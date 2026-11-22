"""Evaluation-only manufacturer loader; portable bundles are explicit and hash checked.

Neither weights nor wrappers belong to the synchronised source/training-input tree.
The default remains the workstation wrapper. No training module imports this loader.
"""
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path

from r1wb.eval.paths import ROOT

_LOCAL = ROOT.parent / ".agent" / "tests" / "r1wb" / "manufacturer_policy.py"


def inspect_bundle(bundle):
  """Validate schema, containment and every digest before executing wrapper code.

  manifest.json: {schema: "r1wb-manufacturer-eval-v1", purpose: "evaluation-only",
  files: {wrapper, weights, source_onnx, provenance: {path: "relative", sha256: "..."}}}.
  The provenance file must bind the source ONNX and converted checkpoint hashes.
  """
  path = Path(bundle).expanduser().resolve()
  if path.is_dir():
    path /= "manifest.json"
  manifest = json.loads(path.read_text())
  if manifest.get("schema") != "r1wb-manufacturer-eval-v1" or manifest.get("purpose") != "evaluation-only":
    raise ValueError("Not an evaluation-only manufacturer bundle")
  entries = manifest.get("files", {})
  if not {"wrapper", "weights", "source_onnx", "provenance"} <= entries.keys():
    raise ValueError("Manufacturer bundle is missing required artifacts")
  paths, assets = {}, []
  for role, entry in entries.items():
    relative = Path(entry["path"])
    file = (path.parent / relative).resolve()
    if relative.is_absolute() or not file.is_relative_to(path.parent) or not file.is_file():
      raise ValueError(f"Manufacturer bundle artifact escapes or is missing: {role}")
    with file.open("rb") as stream:
      digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != entry.get("sha256"):
      raise ValueError(f"Manufacturer bundle digest mismatch: {role}")
    paths[role] = file
    assets.append(dict(role=role, source=str(file), sha256=digest))
  provenance = json.loads(paths["provenance"].read_text())
  if (provenance.get("checkpoint_sha256") != entries["weights"]["sha256"] or
      provenance.get("source_sha256") != entries["source_onnx"]["sha256"]):
    raise ValueError("Manufacturer provenance does not bind the supplied weights and source")
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  return paths, dict(manifest=str(path), manifest_sha256=digest, purpose="evaluation-only", assets=assets)


def load(bundle=None, *, reset_history=False):
  """(weights path, policy callable), preserving the default workstation interface."""
  paths, metadata = inspect_bundle(bundle) if bundle is not None else ({}, None)
  wrapper = paths.get("wrapper", _LOCAL)
  if not wrapper.is_file():
    raise RuntimeError("The manufacturer comparison runs on the workstation only (its wrapper and weights are not synchronised)")
  spec = importlib.util.spec_from_file_location("manufacturer_policy", wrapper)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  weights = paths.get("weights", module.MANUFACTURER)
  policy = partial(module.ManufacturerPolicy, weights_path=weights, reset_history=reset_history)
  policy.bundle_metadata = metadata
  policy.wrapper_path = wrapper
  return weights, policy
