#!/usr/bin/env bash
# IID 联邦全局模型评估 —— 守护式自动触发。
#
# 背景：
#   - 异构联邦 50 轮已完成并补评 task1/task2/task3（eval_hetero_rest.sh）。
#   - IID 联邦 50 轮是门2 的对照组，run_fedavg.sh 跑完只打印"下一步手动评估"，
#     这里用守护脚本自动接上，避免人盯着。
#
# 触发顺序（严格串行，避免 GPU/CPU 争抢）：
#   1) 等 IID 第 50 轮聚合产物 global_r49/model.pth 出现
#   2) 等当前所有 hpt.run_eval（异构补评）进程退出
#   3) 逐任务评估 task1/task2/task3，3 seeds × 20 rollouts
#
# n_jobs=3：3090 只占 <1GB 显存，15 核 CPU，3 个 seed 并行比串行快 ~3 倍。
#           不影响统计口径（每个 seed 独立）。
set -u

SCRIPT_DIR=/root/HPT/federated
GLOBAL=/root/HPT/output/fedavg_iid/global_r49
LOG_TAG=iid_r49

echo "[watch] 等待 IID 50 轮产物 $GLOBAL/model.pth ..."
while [ ! -f "$GLOBAL/model.pth" ]; do sleep 60; done
echo "[watch] global_r49 就绪: $(date)"

echo "[watch] 等待其它评估进程退出（异构补评）..."
while pgrep -f "hpt\.run_eval" >/dev/null 2>&1; do sleep 30; done
echo "[watch] GPU 空闲: $(date)"

for pair in "mujoco_metaworld_task1 reach-v2" \
            "mujoco_metaworld_task2 push-v2" \
            "mujoco_metaworld_task3 button-press-topdown-v2"; do
  set -- $pair
  MODEL_DIR="$GLOBAL" DOMAIN="$1" ENVNAME="$2" \
    N_SEEDS=3 EPISODE_NUM=20 N_JOBS=3 LOG_TAG="$LOG_TAG" \
    bash "$SCRIPT_DIR/eval_model.sh"
done

echo "===== IID 联邦三任务汇总 ====="
for t in 1 2 3; do
  log="/root/HPT/output/eval_${LOG_TAG}_mujoco_metaworld_task${t}.log"
  if [ -f "$log" ]; then
    echo "--- task${t} ---"
    /root/miniconda3/envs/hpt/bin/python "$SCRIPT_DIR/summarize_eval.py" "$log" 2>/dev/null | tail -5
  else
    echo "--- task${t}: 日志缺失 $log ---"
  fi
done
echo "IID_EVAL_DONE"
