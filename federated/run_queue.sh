#!/usr/bin/env bash
# FedVLA：等待当前 GPU 任务空闲后，串行跑完剩余实验队列。
#
# 队列（按优先级）：
#   1. 等预算集中式基线（3 任务）      —— 论文对比表第二列
#   2. 集中式基线评估（3 任务 × 3 seeds × 20 rollouts）
#   3. IID 健全性检查（4 客户端全 task1，50 轮）—— 验证管线正确性
#   4. IID 评估
#
# 用法：
#   nohup bash run_queue.sh > /root/HPT/output/queue.log 2>&1 &
set -u
SCRIPT_DIR=/root/HPT/federated

export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# 等 GPU 上没有 hpt.run 训练进程（最多等 3 小时）
wait_gpu () {
  local max_wait=${1:-10800}
  local waited=0
  while [ "$waited" -lt "$max_wait" ]; do
    n=$(ps -eo cmd | grep -E 'hpt\.run ' | grep -v grep | wc -l)
    if [ "$n" -eq 0 ]; then
      echo "[queue] GPU 空闲（第 ${waited}s）"
      return 0
    fi
    echo "[queue] 等待 GPU 空闲，还有 $n 个训练进程（已等 ${waited}s）"
    sleep 60
    waited=$((waited + 60))
  done
  echo "[queue] 等待超时，继续执行"
  return 0
}

echo "==================== 队列开始 $(date '+%H:%M:%S') ===================="

echo ">>> [1/4] 等预算集中式基线"
wait_gpu
# 已有产物会自动 SKIP（run_centralized_equal_budget.sh 内有 model.pth 存在检查）
bash "$SCRIPT_DIR/run_centralized_equal_budget.sh"

echo ">>> [2/4] 集中式基线评估"
CENTRAL_ROOT=/root/HPT/output/centralized bash "$SCRIPT_DIR/eval_centralized.sh"

echo ">>> [2b/4] 等预算集中式评估"
CENTRAL_ROOT=/root/HPT/output/centralized_equal_budget bash "$SCRIPT_DIR/eval_centralized.sh"

echo ">>> [3/4] IID 健全性检查（4 客户端全 task1）"
wait_gpu
bash "$SCRIPT_DIR/run_all_experiments.sh" iid

echo ">>> [4/4] 生成对比表"
bash "$SCRIPT_DIR/compare_results.sh"

echo "==================== 队列完成 $(date '+%H:%M:%S') ===================="
echo "QUEUE_DONE"
