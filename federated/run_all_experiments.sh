#!/usr/bin/env bash
# FedVLA 正式实验编排：一次跑完 门2 异构 FedAvg + 集中式基线 + 统一评估。
#
# 背景（2026-11-03 已全部实测验证）：
#   - 门0/门1 已过（trunk 键名、本地续训生效）
#   - 门2 烟雾测试已过：2 轮 × 4 客户端，续训自检 max_diff=3.1e-04 > 0
#     全局 trunk 与 4 客户端均值偏差 = 0.000e+00（真等权平均）
#   - epoch_size 实测 task1=15 / task2=19 / task3=24，与探针预测一致
#
# 用法：
#   bash run_all_experiments.sh hetero   # 异构 FedAvg（正式，50 轮）
#   bash run_all_experiments.sh iid      # IID 健全性检查（4 客户端全 task1）
#   bash run_all_experiments.sh eval     # 只做评估
set -u

export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

SCRIPT_DIR=/root/HPT/federated
MODE=${1:-hetero}

# 客户端任务分配
declare -A DOMAINS_HET=( [0]="mujoco_metaworld_task1" [1]="mujoco_metaworld_task2" [2]="mujoco_metaworld_task3" [3]="mujoco_metaworld_task1" )
declare -A ENVS_HET=(   [0]="reach-v2" [1]="push-v2" [2]="button-press-topdown-v2" [3]="reach-v2" )
declare -A EPOCH_HET=(  [0]="15" [1]="19" [2]="24" [3]="15" )
declare -A DOMAINS_IID=( [0]="mujoco_metaworld_task1" [1]="mujoco_metaworld_task1" [2]="mujoco_metaworld_task1" [3]="mujoco_metaworld_task1" )
declare -A ENVS_IID=(   [0]="reach-v2" [1]="reach-v2" [2]="reach-v2" [3]="reach-v2" )
declare -A EPOCH_IID=(  [0]="15" [1]="15" [2]="15" [3]="15" )

case "$MODE" in
  hetero)
    OUT=/root/HPT/output/fedavg_hetero
    D_OVR="${DOMAINS_HET[0]} ${DOMAINS_HET[1]} ${DOMAINS_HET[2]} ${DOMAINS_HET[3]}"
    E_OVR="${ENVS_HET[0]} ${ENVS_HET[1]} ${ENVS_HET[2]} ${ENVS_HET[3]}"
    S_OVR="${EPOCH_HET[0]} ${EPOCH_HET[1]} ${EPOCH_HET[2]} ${EPOCH_HET[3]}"
    ;;
  iid)
    OUT=/root/HPT/output/fedavg_iid
    D_OVR="${DOMAINS_IID[0]} ${DOMAINS_IID[1]} ${DOMAINS_IID[2]} ${DOMAINS_IID[3]}"
    E_OVR="${ENVS_IID[0]} ${ENVS_IID[1]} ${ENVS_IID[2]} ${ENVS_IID[3]}"
    S_OVR="${EPOCH_IID[0]} ${EPOCH_IID[1]} ${EPOCH_IID[2]} ${EPOCH_IID[3]}"
    ;;
  eval)
    OUT=/root/HPT/output/fedavg_hetero
    ;;
  *)
    echo "[!!] 未知模式：$MODE（可选 hetero / iid / eval）"; exit 1 ;;
esac

if [ "$MODE" != "eval" ]; then
  echo "===== FedVLA 正式实验：$MODE  输出 -> $OUT ====="
  echo "domains: $D_OVR"
  echo "envs:    $E_OVR"
  START_ROUND=0 NUM_ROUNDS=50 OUTPUT_ROOT="$OUT" \
    CLIENT_DOMAINS_OVERRIDE="$D_OVR" \
    CLIENT_ENVS_OVERRIDE="$E_OVR" \
    CLIENT_EPOCH_SIZE_OVERRIDE="$S_OVR" \
    bash "$SCRIPT_DIR/run_fedavg.sh" 2>&1 | tee "$OUT/../fedavg_${MODE}_run.log"
  echo "===== 联邦训练结束 ====="
fi

# 评估最终全局模型（3 seeds × 20 rollouts）
GLOBAL="$OUT/global_r49"
if [ ! -f "$GLOBAL/model.pth" ]; then
  echo "[!!] 找不到最终全局模型：$GLOBAL/model.pth"; exit 1
fi

echo "===== 评估最终全局模型 $GLOBAL ====="
# ⚠️ LOG_TAG 必须区分 hetero / iid —— 两者的 MODEL_DIR basename 都是 global_r49，
#    日志同名会互相覆盖（踩过：IID 评估把异构评估结果冲掉）。
if [ "$MODE" = "iid" ]; then
  # IID 只训了 task1 —— 全局模型里没有 task2/task3 的 stem/head，
  # 评估那两个任务没有意义（strict=False 会退化成随机 head）。
  EVAL_PAIRS="mujoco_metaworld_task1 reach-v2"
else
  EVAL_PAIRS="mujoco_metaworld_task1 reach-v2
              mujoco_metaworld_task2 push-v2
              mujoco_metaworld_task3 button-press-topdown-v2"
fi

for pair in $EVAL_PAIRS; do
  set -- $pair
  MODEL_DIR="$GLOBAL" DOMAIN="$1" ENVNAME="$2" N_SEEDS=3 EPISODE_NUM=20 N_JOBS=1 \
    LOG_TAG="${MODE}_r49" \
    bash "$SCRIPT_DIR/eval_model.sh"
done

echo "===== 全部完成 ====="
echo "RUN_ALL_DONE"
