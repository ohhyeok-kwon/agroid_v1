"""Where the code and the work area are: workstation agroid_v1/trainer + agroid_v1/.agent/trainer, GPU server
/rl-agent + /rl-agent/.agent (the same rule as scripts/00_initialize.sh)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                                   # trainer/ or /rl-agent
AGENT = ROOT / ".agent" if (ROOT / ".agent" / "runs").is_dir() else ROOT.parent / ".agent" / "trainer"
ON_SERVER = AGENT.parent == ROOT
