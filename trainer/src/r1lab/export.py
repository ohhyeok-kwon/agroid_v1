"""Export an rsl_rl actor to the agroid ONNX contract and verify it."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from r1lab import contract as C


def export_onnx(policy, path: str | Path, opset: int = 11, verify: bool = True,
                observation_contract: dict | None = None, metadata: dict | None = None) -> Path:
  """Write a fixed-shape actor with its normalizer and observation contract.

  Existing robot policies use 810 inputs. Experimental terrain-aware policies
  use 927; their extra 117 samples must be supplied by the simulation runtime.
  metadata: extra "r1lab.*" string properties (e.g. a distilled student's teacher
  run, checkpoint hash and command contract); it cannot replace contract keys.
  """
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  model = policy.as_onnx(verbose=False)
  model.to("cpu").eval()
  dummy = model.get_dummy_inputs()[0]
  dimension = int(dummy.shape[-1])
  teacher = observation_contract is not None
  if teacher:
    if (observation_contract.get("name") != "reference-teacher-v1"
        or observation_contract.get("dimension") != dimension
        or len(observation_contract.get("reference_sha256", "")) != 64):
      raise ValueError("Invalid reference teacher observation contract")
  elif dimension not in (C.OBS_DIM, C.OBS_DIM+117):
    raise ValueError(f"Unsupported R1 observation size: {dimension}")
  torch.onnx.export(model, (dummy,), str(path), export_params=True, opset_version=opset,
                    input_names=["obs"], output_names=["actions"], dynamic_axes={},
                    dynamo=False)
  import onnx
  exported = onnx.load(str(path))
  properties = {
      "r1lab.observation": (observation_contract["name"] if teacher else
                            "proprioception-v1" if dimension == C.OBS_DIM else "proprioception-height-v1"),
      "r1lab.input_dimension": str(dimension),
      "r1lab.deployment": "robot-compatible" if dimension == C.OBS_DIM and not teacher else "simulation-only",
  }
  if teacher:
    properties.update({"r1lab.reference_required": "true",
                     "r1lab.reference_sha256": observation_contract["reference_sha256"],
                     "r1lab.phase": observation_contract.get("phase_mode", "clip_fraction")})
  extra = {str(k): str(v) for k, v in (metadata or {}).items()}
  if any(not k.startswith("r1lab.") or k in properties for k in extra):
    raise ValueError("Extra ONNX metadata must use new r1lab.* keys")
  properties.update(extra)
  onnx.helper.set_model_props(exported, properties)
  onnx.save(exported, str(path))
  if verify:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(1, dimension)).astype(np.float32)
    y_onnx = sess.run(["actions"], {"obs": x})[0]
    with torch.no_grad():
      y_t = model(torch.from_numpy(x)).numpy()
    err = float(np.abs(y_onnx - y_t).max())
    assert y_onnx.shape == (1, C.NUM_DOF), y_onnx.shape
    assert err < 1e-4, err
  return path
