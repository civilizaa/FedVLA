#!/usr/bin/env bash
# 决定性诊断：评估等预算集中式的 task1（2250 步 = 联邦客户端的同量训练）
# 目的：区分两种可能
#   (A) 联邦成功率低纯粹因为训练量少（150 epoch vs 400 epoch）
#       -> 等预算集中式也会是低成功率
#   (B) 联邦有额外问题（异构有害 / 聚合 bug）
#       -> 等预算集中式成功率明显高于联邦
set -u
export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

MODEL=/root/HPT/output/centralized_equal_budget/task1
echo "模型: $MODEL"
MODEL_DIR="$MODEL" DOMAIN=mujoco_metaworld_task1 ENVNAME=reach-v2 \
LOG_TAG=diag_equalbudget_task1 \
N_SEEDS=1 EPISODE_NUM=20 N_JOBS=1 \
  bash /root/HPT/federated/eval_model.sh

echo "===== 汇总 ====="
/root/miniconda3/envs/hpt/bin/python /root/HPT/federated/summarize_eval.py \
  /root/HPT/output/eval_diag_equalbudget_task1_mujoco_metaworld_task1.log 2>/dev/null | tail -8
echo "DIAG_DONE"
