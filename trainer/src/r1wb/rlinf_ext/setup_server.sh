#!/usr/bin/env bash
# Trial setup in the server's work area: registers the "r1wb" environment type in an RLinf checkout (two small
# patches: the type and its action pass-through) and links this folder's configs into RLinf's config directory.
#   src/r1wb/rlinf_ext/setup_server.sh [RLinf checkout, default .agent/rlinf/RLinf]      (run in /rl-agent)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="${1:-.agent/rlinf/RLinf}"
python3 - "$repo" <<'PY'
import sys
from pathlib import Path
repo = Path(sys.argv[1])
p = repo / "rlinf/envs/__init__.py"; s = p.read_text()
if 'R1WB = "r1wb"' not in s:
    s = s.replace('    POLARIS = "polaris"\n', '    POLARIS = "polaris"\n    R1WB = "r1wb"\n', 1)
    marker = "    elif env_type == SupportedEnvType.GENESIS:\n"
    assert marker in s
    s = s.replace(marker, "    elif env_type == SupportedEnvType.R1WB:\n        from r1wb.rlinf_ext.env import R1TaskEnv\n\n        return R1TaskEnv\n" + marker, 1)
    p.write_text(s)
p = repo / "rlinf/envs/action_utils.py"; s = p.read_text()
if "SupportedEnvType.R1WB" not in s:
    marker = "    elif env_type == SupportedEnvType.GENESIS:\n        chunk_actions = prepare_actions_for_genesis("
    assert marker in s
    s = s.replace(marker, "    elif env_type == SupportedEnvType.R1WB:\n        chunk_actions = torch.from_numpy(raw_chunk_actions).float()\n" + marker, 1)
    p.write_text(s)
print("environment type registered in", repo)
PY
cfg="$repo/examples/embodiment/config"
ln -sfn "$here/config/r1wb_task_ppo_mlp.yaml" "$cfg/r1wb_task_ppo_mlp.yaml"
ln -sfn "$here/config/r1wb_task_ppo_mlp_2gpu.yaml" "$cfg/r1wb_task_ppo_mlp_2gpu.yaml"
ln -sfn "$here/config/r1wb_task_ppo_mlp_actor2.yaml" "$cfg/r1wb_task_ppo_mlp_actor2.yaml"
ln -sfn "$here/config/env/r1wb_task.yaml" "$cfg/env/r1wb_task.yaml"
echo "configs linked into $cfg"
