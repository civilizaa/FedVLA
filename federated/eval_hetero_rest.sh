#!/usr/bin/env bash
# 评估异构联邦全局模型在 task2 / task3 上的表现（task1 已有）。
#
# 背景：门2 判据需要"集中式 vs IID联邦 vs 异构联邦"的完整三任务对比，
# 而异构 global_r49 目前只评过 task1（= 0.05）。
#
# 用法：bash eval_hetero_rest.sh
set -u

SCRIPT_DIR=/root/HPT/federated
GLOBAL=/root/HPT/output/fedavg_hetero/global_r49
LOG_TAG=hetero_r49

if [ ! -f "$GLOBAL/model.pth" ]; then
  echo "[!!] 找不到全局模型：$GLOBAL/model.pth"; exit 1
fi

echo "===== 异构联邦全局模型：补评 task2 / task3 ====="
for pair in "mujoco_metaworld_task2 push-v2" \
            "mujoco_metaworld_task3 button-press-topdown-v2"; do
  set -- $pair
  MODEL_DIR="$GLOBAL" DOMAIN="$1" ENVNAME="$2" \
  N_SEEDS=3 EPISODE_NUM=20 N_JOBS=1 LOG_TAG="$LOG_TAG" \
    bash "$SCRIPT_DIR/eval_model.sh"
done

echo "===== 异构联邦三任务汇总 ====="
for t in 1 2 3; do
  log="/root/HPT/output/eval_${LOG_TAG}_mujoco_metaworld_task${t}.log"
  if [ -f "$log" ]; then
    echo "--- task${t} ---"
    /root/miniconda3/envs/hpt/bin/python "$SCRIPT_DIR/summarize_eval.py" "$log" 2>/dev/null | tail -5
  else
    echo "--- task${t}: 日志缺失 $log ---"
  fi
done
echo "EVAL_HETERO_REST_DONE"
