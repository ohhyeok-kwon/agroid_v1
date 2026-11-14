"""A run folder that holds a checkpoint's policy averaged with its own mirror image (r1wb.symmetry.Symmetrized).

  policy_s(o) = (policy(o) + mirror(policy(mirror(o)))) / 2

Nothing is trained: the weights are the source checkpoint's, a flag in the new checkpoint tells
student.observations.load_teacher to apply the average, and policy_<iteration>.onnx contains the whole function
(two passes of the network), so every tool that takes "run folder + iteration" uses the symmetric policy.
usage: python -m r1wb.symmetrize <source run folder> <iteration> <run name>      (writes next to the source run)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


class Exported(torch.nn.Module):
  """The symmetric policy as one function of the observation tensor (for ONNX)."""

  def __init__(self, model, mirror):
    super().__init__()
    self.model = model
    self.register_buffer("obs_index", mirror.obs_index.cpu())
    self.register_buffer("obs_sign", mirror.obs_sign.cpu())
    self.register_buffer("act_index", mirror.act_index.cpu())
    self.register_buffer("act_sign", mirror.act_sign.cpu())

  def one(self, obs):
    return self.model.mlp(self.model.obs_normalizer(obs))

  def forward(self, obs):
    flipped = torch.index_select(obs, 1, self.obs_index) * self.obs_sign
    return 0.5 * (self.one(obs) + torch.index_select(self.one(flipped), 1, self.act_index) * self.act_sign)


def main():
  from r1wb import symmetry
  from r1wb.student.observations import load_teacher
  source, iteration, name = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
  info = json.loads((source / "run_info.json").read_text())
  stamp = time.strftime("%Y%m%d-%H%M")
  run_id = f"{info['action']}_{stamp}_{info['tags']}-symmetrized_n{info['num_envs']}_{name}"
  target = source.parent / run_id
  target.mkdir()
  loaded = torch.load(source / f"model_{iteration}.pt", map_location="cpu", weights_only=False)
  loaded["symmetrized"] = True
  torch.save(loaded, target / f"model_{iteration}.pt")
  if (source / "params").is_dir():
    shutil.copytree(source / "params", target / "params")
  plain = load_teacher(str(source / f"model_{iteration}.pt"), "cpu")
  both = load_teacher(str(target / f"model_{iteration}.pt"), "cpu")
  mirror = symmetry.ActorMirror(plain.obs_dim, "cpu")
  exported = Exported(plain, mirror).eval()
  x = torch.randn(64, plain.obs_dim)
  with torch.no_grad():
    y = both(TensorDict({"actor": x}, batch_size=[64]))
    assert float((exported(x) - y).abs().max()) < 1e-5
    twin = mirror.action(both(TensorDict({"actor": mirror.observation(x)}, batch_size=[64])))
    commutes = float((twin - y).abs().max())
    before = float((mirror.action(plain(TensorDict({"actor": mirror.observation(x)}, batch_size=[64])))
                    - plain(TensorDict({"actor": x}, batch_size=[64]))).abs().mean())
  path = target / f"policy_{iteration}.onnx"
  torch.onnx.export(exported, (torch.zeros(1, plain.obs_dim),), str(path), export_params=True, opset_version=13,
                    input_names=["obs"], output_names=["actions"], dynamic_axes={}, dynamo=False)
  import onnx
  import onnxruntime as ort
  model = onnx.load(str(path))
  onnx.helper.set_model_props(model, {"r1lab.observation": "reference-teacher-task-v1", "r1lab.input_dimension": str(plain.obs_dim),
                                      "r1lab.deployment": "simulation-only", "r1lab.symmetrized_from": f"{info['id']}@{iteration}"})
  onnx.save(model, str(path))
  session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  sample = np.random.default_rng(0).normal(size=(1, plain.obs_dim)).astype(np.float32)
  with torch.no_grad():
    error = float(np.abs(session.run(["actions"], {"obs": sample})[0] - exported(torch.from_numpy(sample)).numpy()).max())
  assert error < 1e-4, error
  info.update(id=run_id, tags=info["tags"] + "-symmetrized", started=stamp, state="completed", iteration=iteration, iters=0,
              description=f"{info['id']}@{iteration} averaged with its own mirror image (r1wb.symmetrize); nothing trained",
              derived_from=dict(run=info["id"], iteration=iteration, method="r1wb.symmetry.Symmetrized"),
              created_at=time.time(), updated_at=time.time(), pid=None, log_file=None, checkpoint=f"model_{iteration}.pt",
              policy=path.name)
  (target / "run_info.json").write_text(json.dumps(info, indent=1))
  print(f"{target}\nmirror test on random inputs: |mirror(policy(mirror o)) - policy(o)| plain {before:.4f} action units, symmetric {commutes:.1e};"
        f" onnx against torch {error:.1e}")


if __name__ == "__main__":
  main()
