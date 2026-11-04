#!/usr/bin/env bash
# 云端恢复后的「门2 收尾」脚本 —— 幂等，可在 agent 复活后反复执行。
#
# 它做三件事：
#   1) 检查异构联邦 global_r49 的 task3 评估是否完成；没完成就只补 task3
#      （eval_hetero_rest.sh 会连 task2 一起重跑，浪费时间，所以单独拆）
#   2) 检查 IID 联邦 global_r49 的三任务评估是否完成；没完成就拉起守护脚本
#   3) 打印全部关键结果（集中式 / 等预算 / 异构联邦 / IID 联邦）
#
# 用法：bash resume_gate2.sh
set -u

# ---- 互斥锁：防止自动化轮询与手动触发同时跑两份评估 ----
LOCK=/root/HPT/output/.resume_gate2.lock
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[skip] 已有 resume_gate2 实例在运行（锁 $LOCK），本次退出。"
  exit 0
fi

SCRIPT_DIR=/root/HPT/federated
HETERO_G=/root/HPT/output/fedavg_hetero/global_r49
IID_G=/root/HPT/output/fedavg_iid/global_r49
HPT_PY=/root/miniconda3/envs/hpt/bin/python

echo "===== [0] 环境体检 ====="
date
df -h / | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
for d in "$HETERO_G" "$IID_G"; do
  if [ -f "$d/model.pth" ]; then
    echo "[OK] $d 存在"
  else
    echo "[!!] 缺失 $d/model.pth"
  fi
done

# ---- 工具：数一个评估日志里已完成的 seed 数（每个 seed 一张 reward 表）----
count_seeds() {
  # $1 = log 路径；每个 seed 结束会打印 "| avg" 行
  [ -f "$1" ] || { echo 0; return; }
  grep -c '^| avg' "$1" 2>/dev/null || echo 0
}

echo
echo "===== [1] 异构联邦 global_r49 补评 ====="
for t in 1 2 3; do
  L="/root/HPT/output/eval_hetero_r49_mujoco_metaworld_task${t}.log"
  echo "  task${t}: 已完成 seed 数 = $(count_seeds "$L")"
done

if [ "$(count_seeds /root/HPT/output/eval_hetero_r49_mujoco_metaworld_task3.log)" -lt 3 ]; then
  echo "[行动] 异构 task3 评估未完成 -> 单独补评（不重跑 task1/task2）"
  MODEL_DIR="$HETERO_G" DOMAIN=mujoco_metaworld_task3 \
    ENVNAME=button-press-topdown-v2 \
    N_SEEDS=3 EPISODE_NUM=20 N_JOBS=3 LOG_TAG=hetero_r49 \
    bash "$SCRIPT_DIR/eval_model.sh"
else
  echo "[跳过] 异构 task3 已完成"
fi

echo
echo "===== [2] IID 联邦 global_r49 评估 ====="
IID_DONE=1
for t in 1 2 3; do
  L="/root/HPT/output/eval_iid_r49_mujoco_metaworld_task${t}.log"
  N=$(count_seeds "$L")
  echo "  task${t}: 已完成 seed 数 = $N"
  [ "$N" -ge 3 ] || IID_DONE=0
done

if [ "$IID_DONE" -eq 0 ]; then
  echo "[行动] IID 评估未完成 -> 拉起 run_iid_eval.sh（前台等待，约 1 小时）"
  bash "$SCRIPT_DIR/run_iid_eval.sh"
else
  echo "[跳过] IID 三任务已全部评估完成"
fi

echo
echo "===== [3] 门2 关键结果汇总 ====="
summ() {  # $1=glob 模式  $2=标签
  local hits
  hits=$(ls -1t /root/HPT/output/eval_$1 2>/dev/null | head -1)
  if [ -n "$hits" ]; then
    echo "--- $2  [$hits] ---"
    "$HPT_PY" "$SCRIPT_DIR/summarize_eval.py" "$hits" 2>/dev/null | grep -E 'n=[0-9]' || echo "(解析失败)"
  else
    echo "--- $2: 未找到 eval_$1 ---"
  fi
}
summ "centralized*task1*.log"      "集中式 task1 (6000步)"
summ "centralized*task2*.log"      "集中式 task2 (6000步)"
summ "centralized*task3*.log"      "集中式 task3 (6000步)"
summ "*equal_budget*task1*.log"    "等预算集中式 task1 (2250步)"
summ "client0_r49*task1*.log"      "异构 client_0_r49 task1"
summ "hetero_r49*task1*.log"       "异构 global_r49 task1"
summ "hetero_r49*task2*.log"       "异构 global_r49 task2"
summ "hetero_r49*task3*.log"       "异构 global_r49 task3"
summ "iid_r49*task1*.log"          "IID global_r49 task1"
summ "iid_r49*task2*.log"          "IID global_r49 task2"
summ "iid_r49*task3*.log"          "IID global_r49 task3"

echo
echo "===== 全部评估日志清单 ====="
ls -1t /root/HPT/output/eval_*.log 2>/dev/null | head -30

echo
echo "===== [4] 磁盘 ====="
df -h / | tail -1
echo "RESUME_GATE2_DONE"
