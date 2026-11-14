#!/usr/bin/env bash
# Interactive shell on the GPU server in the project folder (headless EGL rendering env set).
set -euo pipefail
host="${RL_REMOTE:-super-via-ailab}"
rdir="${RL_REMOTE_DIR:-/rl-agent}"
egl="/opt/nvidia-egl"
exec ssh -t "$host" "cd $rdir && export MUJOCO_GL=egl && if [ -f $egl/10_nvidia.json ]; then export __EGL_VENDOR_LIBRARY_FILENAMES=$egl/10_nvidia.json LD_LIBRARY_PATH=$egl/usr/lib/x86_64-linux-gnu\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}; fi && exec bash"
