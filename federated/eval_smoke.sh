#!/usr/bin/env bash
# 门2 收尾：对烟雾测试的全局模型做评估（每任务 1 seed × 10 rollouts，快速验证可用性）
# 正式评估走 eval_model.sh（3 seeds × 20 rollouts）
set -u
GLOBAL=${1:-/root/HPT/output/smoke_hetero/global_r1}
export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

run () {
  local dom="$1" env="$2"
  echo "########## 评估 $dom ($env) ##########"
  MODEL_DIR="$GLOBAL" DOMAIN="$dom" ENVNAME="$env" N_SEEDS=1 EPISODE_NUM=10 N_JOBS=1 \
    bash /root/HPT/federated/eval_model.sh 2>&1 | grep -iE "success|rc=|rc\]|error|Traceback" | head -20
}

run mujoco_metaworld_task1 reach-v2
run mujoco_metaworld_task2 push-v2
run mujoco_metaworld_task3 button-press-topdown-v2
echo "EVAL_ALL_DONE"
