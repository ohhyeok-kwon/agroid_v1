"""The manufacturer's walking policy is used for comparison only, on the workstation: neither its weights nor the
wrapper that runs them are part of the synchronised source tree. This loads the wrapper from the workstation's
.agent/tests if it is there."""
import importlib.util

from r1wb.eval.paths import ROOT

_LOCAL = ROOT.parent / ".agent" / "tests" / "r1wb" / "manufacturer_policy.py"


def load():
  """(path of the weights, policy class); raises where the comparison is not available (the GPU server)."""
  if not _LOCAL.is_file():
    raise RuntimeError("The manufacturer comparison runs on the workstation only (its wrapper and weights are not synchronised)")
  spec = importlib.util.spec_from_file_location("manufacturer_policy", _LOCAL)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.MANUFACTURER, module.ManufacturerPolicy
