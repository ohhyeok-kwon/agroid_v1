"""Single-policy distributed helpers; the native rsl_rl runner owns group setup."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
  import torch
  from torch import Tensor
  from torch.nn import Module


@dataclass(frozen=True)
class DistributedContext:
  rank: int = 0
  local_rank: int = 0
  world_size: int = 1
  local_world_size: int = 1

  @property
  def device(self) -> str:
    return f"cuda:{self.local_rank}"

  @property
  def is_primary(self) -> bool:
    return self.rank == 0

  @property
  def is_distributed(self) -> bool:
    return self.world_size > 1


def distributed_context(environ: Mapping[str, str] | None = None) -> DistributedContext:
  """Read torchrun ranks without importing torch or allocating GPU resources."""
  environ = os.environ if environ is None else environ

  def integer(name: str, default: int) -> int:
    try:
      return int(environ.get(name, str(default)))
    except ValueError as error:
      raise ValueError(f"{name} must be an integer") from error

  world_size = integer("WORLD_SIZE", 1)
  local_world_size = integer("LOCAL_WORLD_SIZE", world_size)
  rank = integer("RANK", 0)
  local_rank = integer("LOCAL_RANK", 0)
  if not 1 <= local_world_size <= world_size:
    raise ValueError("Require 1 <= LOCAL_WORLD_SIZE <= WORLD_SIZE")
  if not 0 <= rank < world_size:
    raise ValueError("Require 0 <= RANK < WORLD_SIZE")
  if not 0 <= local_rank < local_world_size:
    raise ValueError("Require 0 <= LOCAL_RANK < LOCAL_WORLD_SIZE")
  if world_size > 1 and any(key not in environ for key in ("RANK", "LOCAL_RANK")):
    raise ValueError("Distributed training requires torchrun RANK and LOCAL_RANK")
  return DistributedContext(rank, local_rank, world_size, local_world_size)


def configure_local_device(context: DistributedContext | None = None) -> str:
  """Select this worker's CUDA/EGL device before constructing the MuJoCo env.

  This deliberately does not initialize torch.distributed: rsl_rl does that
  inside its runner, after environment construction.
  """
  context = context or distributed_context()
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(context.local_rank)
  import torch
  torch.cuda.set_device(context.local_rank)
  return context.device


def _process_group():
  import torch.distributed as dist
  if dist.is_available() and dist.is_initialized():
    return dist
  if distributed_context().is_distributed:
    raise RuntimeError("Collectives require the native rsl_rl runner's initialized process group")
  return None


def all_reduce_mean(value: Tensor) -> Tensor:
  """Return a detached rank mean without modifying the input tensor."""
  import torch
  result = value.detach().clone()
  if not (result.is_floating_point() or result.is_complex()):
    result = result.to(torch.float64)
  dist = _process_group()
  if dist is not None and dist.get_world_size() > 1:
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    result.div_(dist.get_world_size())
  return result


def global_moments(samples: Tensor, *, dim: int = 0) -> tuple[Tensor, Tensor, Tensor]:
  """Return global mean, population variance, and scalar int64 sample count.

  Local batches may have different lengths, including zero. Moment reductions
  use float64 and the parallel variance formula; outputs use the sample dtype
  (float32 for integer/half inputs). Call on every rank in the same order.
  """
  import torch
  dist = _process_group()
  if samples.ndim == 0 or not -samples.ndim <= dim < samples.ndim:
    raise ValueError("global_moments requires a valid sample dimension")
  dim %= samples.ndim
  dtype = samples.dtype if samples.dtype in (torch.float32, torch.float64) else torch.float32
  samples = samples.detach().to(dtype)
  count = samples.shape[dim]
  shape = samples.shape[:dim] + samples.shape[dim + 1:]
  if count:
    variance, mean = torch.var_mean(samples, dim=dim, correction=0)
  else:
    mean = torch.zeros(shape, dtype=dtype, device=samples.device)
    variance = torch.zeros_like(mean)
  mean64, variance64 = mean.double(), variance.double()
  weighted = torch.cat((mean64.new_tensor([count]), (mean64 * count).reshape(-1)))
  if dist is not None and dist.get_world_size() > 1:
    dist.all_reduce(weighted, op=dist.ReduceOp.SUM)
  total = weighted[0]
  if total.item() == 0:
    raise ValueError("global_moments received no samples on any rank")
  global_mean = (weighted[1:] / total).reshape(shape)
  m2 = count * (variance64 + (mean64 - global_mean).square())
  if dist is not None and dist.get_world_size() > 1:
    dist.all_reduce(m2, op=dist.ReduceOp.SUM)
  return global_mean.to(dtype), (m2 / total).clamp_min(0).to(dtype), total.to(torch.int64)


def broadcast_module(module: Module, *, src: int = 0) -> None:
  """Broadcast all parameters and buffers after the native group exists."""
  import torch
  dist = _process_group()
  if dist is not None and dist.get_world_size() > 1:
    with torch.no_grad():
      for value in module.state_dict().values():
        dist.broadcast(value, src=src)


def average_gradients(module: Module) -> None:
  """Average dense gradients, including parameters used on only some ranks."""
  import torch
  dist = _process_group()
  if dist is None or dist.get_world_size() == 1:
    return
  parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
  if not parameters:
    return
  present = torch.tensor([parameter.grad is not None for parameter in parameters],
                         dtype=torch.int32, device=parameters[0].device)
  dist.all_reduce(present, op=dist.ReduceOp.SUM)
  active = [parameter for parameter, used in zip(parameters, present.tolist()) if used]
  if not active:
    return
  gradients = [parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
               for parameter in active]
  if any(gradient.is_sparse for gradient in gradients):
    raise ValueError("average_gradients supports dense gradients only")
  flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
  dist.all_reduce(flat, op=dist.ReduceOp.SUM)
  flat.div_(dist.get_world_size())
  offset = 0
  for parameter in active:
    size = parameter.numel()
    gradient = flat[offset:offset + size].view_as(parameter).to(parameter.dtype)
    if parameter.grad is None:
      parameter.grad = gradient.clone()
    else:
      parameter.grad.copy_(gradient)
    offset += size


def model_checksum(module: Module, *, include_buffers: bool = False) -> str:
  """Hash parameter names, shapes, dtypes, and exact bytes for run evidence."""
  import torch
  values = list(module.named_parameters())
  if include_buffers:
    values.extend(module.named_buffers())
  digest = hashlib.sha256()
  for name, value in sorted(values):
    digest.update(repr((name, tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
  return digest.hexdigest()


def assert_model_consistent(module: Module, *, include_buffers: bool = False,
                            label: str = "model") -> str:
  """Raise on every rank if model checksums disagree; return the shared hash."""
  dist = _process_group()
  checksum = model_checksum(module, include_buffers=include_buffers)
  if dist is not None and dist.get_world_size() > 1:
    checksums = [None] * dist.get_world_size()
    dist.all_gather_object(checksums, checksum)
    if len(set(checksums)) != 1:
      detail = ", ".join(f"{rank}:{value[:12]}" for rank, value in enumerate(checksums))
      raise RuntimeError(f"{label} differs across distributed ranks ({detail})")
  return checksum
