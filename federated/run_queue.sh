#!/usr/bin/env bash
# FedVLA：剩余实验队列（幂等版）。
#
# 幂等保证：每一步都先检查产物是否已存在，存在就跳过。
#   - 训练：model.pth 存在 -> SKIP
#   - 评估：eval_centralized_<task>.log 里有 3 张结果表 -> SKIP
#   - IID：global_r49 存在 -> SKIP
#
# 同时检测 hpt.run 和 hpt.run_eval 两种进程，避免与正在跑的评估争 GPU。
set -u
SCRIPT_DIR=/root/HPT/federated
HPT_PY=/root/miniconda3/envs/hpt/bin/python

export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# 等 GPU 空闲（训练和评估都不算占用）
wait_gpu () {
  local max_wait=${1:-36000}
  local waited=0
  while [ "$waited" -lt "$max_wait" ]; do
    n=$(ps -eo cmd | grep -E 'hpt\.run |hpt\.run_eval' | grep -v grep | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "[queue] GPU 空闲（第 ${waited}s）"
      return 0
    fi
    if [ $((waited % 300)) -eq 0 ]; then
      echo "[queue] 等待 GPU 空闲，还有 $n 个进程（已等 ${waited}s）"
    fi
    sleep 60
    waited=$((waited + 60))
  done
  echo "[queue] 等待超时，继续执行"
  return 0
}

# 评估是否已完成（3 张结果表 = 3 seeds）
eval_done () {
  local log="$1"
  [ -f "$log" ] || return 1
  local n
  n=$(grep -c '| avg' "$log" 2>/dev/null || echo 0)
  [ "$n" -ge 3 ]
}

echo "==================== 队列开始 $(date '+%H:%M:%S') ===================="

echo ">>> [1/5] 等预算集中式基线"
wait_gpu
bash "$SCRIPT_DIR/run_centralized_equal_budget.sh"

echo ">>> [2/5] 集中式基线评估（6000 步）"
for t in task1 task2 task3; do
  log="/root/HPT/output/eval_centralized_${t}.log"
  if eval_done "$log"; then
    echo "[SKIP] $t 评估已完成：$log"
  else
    wait_gpu
    CENTRAL_ROOT=/root/HPT/output/centralized bash "$SCRIPT_DIR/eval_centralized.sh"
    break   # eval_centralized.sh 一次跑完三个任务
  fi
done

echo ">>> [3/5] 等预算集中式评估"
for t in task1 task2 task3; do
  log="/root/HPT/output/eval_equalbudget_${t}.log"
  if eval_done "$log"; then
    echo "[SKIP] $t 等预算评估已完成"
  else
    wait_gpu
    CENTRAL_ROOT=/root/HPT/output/centralized_equal_budget \
      ROOT_TAG=centralized_equal_budget \
      ARCHIVE_PREFIX=equalbudget \
      bash "$SCRIPT_DIR/eval_centralized.sh"
    break
  fi
done

echo ">>> [4/5] IID 健全性检查（4 客户端全 task1）"
if [ -f /root/HPT/output/fedavg_iid/global_r49/model.pth ]; then
  echo "[SKIP] IID 已训完"
else
  wait_gpu
  bash "$SCRIPT_DIR/run_all_experiments.sh" iid
fi

echo ">>> [5/5] 生成对比表"
bash "$SCRIPT_DIR/compare_results.sh"

echo "==================== 队列完成 $(date '+%H:%M:%S') ===================="
echo "QUEUE_DONE"
