#!/usr/bin/env bash
# FedVLA：集中式基线的统一评估（与联邦评估同一口径）
#
# 输出约定（compare_results.sh 依赖）：
#   模型目录: /root/HPT/output/centralized/<task>/
#   日志:     /root/HPT/output/eval_centralized_<task>.log
set -u
SCRIPT_DIR=/root/HPT/federated
CENTRAL_ROOT=${CENTRAL_ROOT:-/root/HPT/output/centralized}
N_SEEDS=${N_SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-1}

export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

TASKS=(
  "task1 mujoco_metaworld_task1 reach-v2"
  "task2 mujoco_metaworld_task2 push-v2"
  "task3 mujoco_metaworld_task3 button-press-topdown-v2"
)

for t in "${TASKS[@]}"; do
  set -- $t
  task="$1"; dom="$2"; env="$3"
  model="$CENTRAL_ROOT/$task"
  if [ ! -f "$model/model.pth" ]; then
    echo "[SKIP] $task 没有模型：$model/model.pth"
    continue
  fi
  echo "########## 集中式评估 $task ($env) ##########"
  # eval_model.sh 会写 output/eval_<TAG>_<DOMAIN>.log，这里再拷一份成
  # eval_centralized_<task>.log 作为对比脚本的稳定入口
  MODEL_DIR="$model" DOMAIN="$dom" ENVNAME="$env" \
  N_SEEDS="$N_SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    bash "$SCRIPT_DIR/eval_model.sh"
  src="/root/HPT/output/eval_$(basename "$model")_${dom}.log"
  if [ -f "$src" ]; then
    cp "$src" "/root/HPT/output/eval_centralized_${task}.log"
    echo "[OK] 结果已归档 -> /root/HPT/output/eval_centralized_${task}.log"
  fi
done
echo "CENTRAL_EVAL_DONE"
