#!/usr/bin/env bash
set -eo pipefail

CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
PYTHON="$HOME/miniconda3/envs/anygrasp_py38/bin/python3"
ONLINE_NODE="$HOME/piper_ros/src/piper_anygrasp_demo/scripts/anygrasp_online_inference_node.py"

test -f "$CONDA_SH" || { echo "[错误] 找不到Conda初始化脚本：$CONDA_SH" >&2; exit 1; }
test -x "$PYTHON" || { echo "[错误] 找不到AnyGrasp Python：$PYTHON" >&2; exit 1; }
test -f "$ONLINE_NODE" || { echo "[错误] 找不到在线推理节点：$ONLINE_NODE" >&2; exit 1; }

source /opt/ros/noetic/setup.bash
source "$HOME/gazebo_link_attacher_ws/devel/setup.bash"
source "$HOME/piper_ros/devel/setup.bash"
source "$CONDA_SH"
conda activate anygrasp_py38

unset PYTHONHOME
unset PYTHONNOUSERSITE
export PYTHONPATH="/opt/ros/noetic/lib/python3/dist-packages:$HOME/anygrasp_ws/anygrasp_sdk/grasp_detection:$HOME/anygrasp_ws/pointnet2/build/lib.linux-x86_64-cpython-38:$HOME/gazebo_link_attacher_ws/devel/lib/python3/dist-packages:$HOME/piper_ros/devel/lib/python3/dist-packages"

cd "$HOME/anygrasp_ws/anygrasp_sdk/grasp_detection"
exec "$PYTHON" "$ONLINE_NODE" "$@"
