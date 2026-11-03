#!/usr/bin/env bash
# FedVLA：集中式基线的统一评估（与联邦评估同一口径）
#
# ⚠️ 日志命名必须区分不同基线，否则会互相覆盖：
#    两种基线的模型目录 basename 都是 task1/task2/task3
#    （output/centralized/task1 vs output/centralized_equal_budget/task1），
#    若都用 eval_task1_<domain>.log / eval_centralized_task1.log，
#    第二种基线会把第一种的结果覆盖掉。
#
# 命名规则：
#   原始日志: eval_<ROOT_TAG>_<task>_<domain>.log
#   归档:     eval_<ARCHIVE_PREFIX>_<task>.log
#   ROOT_TAG 默认取 CENTRAL_ROOT 的 basename（如 centralized / centralized_equal_budget）
#   ARCHIVE_PREFIX 默认同 ROOT_TAG
set -u
SCRIPT_DIR=/root/HPT/federated
CENTRAL_ROOT=${CENTRAL_ROOT:-/root/HPT/output/centralized}
N_SEEDS=${N_SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-1}

ROOT_TAG=${ROOT_TAG:-$(basename "$CENTRAL_ROOT")}
ARCHIVE_PREFIX=${ARCHIVE_PREFIX:-$ROOT_TAG}

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
  echo "########## 集中式评估 $task ($env)  root=$ROOT_TAG ##########"

  # eval_model.sh 会写 output/eval_<LOG_TAG>_<domain>.log，
  # 两种基线的模型目录 basename 相同 -> 用 LOG_TAG 区分开。
  RAW="/root/HPT/output/eval_${ROOT_TAG}_${task}_${dom}.log"

  MODEL_DIR="$model" DOMAIN="$dom" ENVNAME="$env" \
  LOG_TAG="${ROOT_TAG}_${task}" \
  N_SEEDS="$N_SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    bash "$SCRIPT_DIR/eval_model.sh"

  # eval_model.sh 实际写的是 eval_<LOG_TAG>_<domain>.log
  produced="/root/HPT/output/eval_${ROOT_TAG}_${task}_${dom}.log"
  if [ -f "$produced" ]; then
    cp "$produced" "$RAW"
    cp "$RAW" "/root/HPT/output/eval_${ARCHIVE_PREFIX}_${task}.log"
    echo "[OK] 结果已归档 -> /root/HPT/output/eval_${ARCHIVE_PREFIX}_${task}.log"
  else
    echo "[!!] 没找到评估输出：$produced"
  fi
done
echo "CENTRAL_EVAL_DONE"
