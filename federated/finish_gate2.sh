#!/usr/bin/env bash
# 门2 收尾 part 2 —— 把所有"1 seed / 未跑"的评估补齐到 3 seeds × 20 rollouts。
#
# 背景（2026-11-04）：
#   云端实例关机重启后，resume_gate2.sh 因 grep -c 退出码 bug 把异构 task3
#   误判成"已完成"而跳过；同时异构 task1 / client_0_r49 / 等预算集中式
#   都只跑了 1 个 seed，不满足论文统计底线（3 seeds × 20 rollouts）。
#   本脚本补齐这些缺口。
#
# 顺序：等 IID 三任务评估跑完 -> 逐个补评 -> 打印最终汇总
set -u

LOCK=/root/HPT/output/.finish_gate2.lock
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[skip] finish_gate2 已有实例在运行（锁 $LOCK），本次退出。"
  exit 0
fi

SCRIPT_DIR=/root/HPT/federated
HETERO_G=/root/HPT/output/fedavg_hetero/global_r49
HETERO_C0=/root/HPT/output/fedavg_hetero/client_0_r49
EQBUDGET=/root/HPT/output/centralized_equal_budget/task1
HPT_PY=/root/miniconda3/envs/hpt/bin/python

# count_seeds：数一个评估日志里已完成的 seed 数（每个 seed 结束打印一行 "| avg"）
# ⚠️ grep -c 在 0 匹配时退出码为 1，不能直接用 `|| echo 0`（会变成 "0\n0"）。
count_seeds() {
  local n
  n=$(grep -c '^| avg' "$1" 2>/dev/null)
  [ -n "$n" ] || n=0
  echo "$n"
}

echo "[finish] 启动于 $(date)"
echo "[finish] 等待所有 hpt.run_eval 结束（IID 三任务评估在跑）..."
while pgrep -f "hpt\.run_eval" >/dev/null 2>&1; do sleep 30; done
echo "[finish] 评估槽位空闲: $(date)"

run_eval() {  # $1=模型目录 $2=domain $3=env $4=LOG_TAG
  local dir="$1" dom="$2" env="$3" tag="$4"
  local log="/root/HPT/output/eval_${tag}_${dom}.log"
  local n; n=$(count_seeds "$log")
  if [ "$n" -ge 3 ]; then
    echo "[finish] $tag $dom 已完成 $n 个 seed —— 跳过"
    return
  fi
  echo "[finish] 评估 $tag $dom ($env)，已完成 seed=$n -> 补到 3"
  MODEL_DIR="$dir" DOMAIN="$dom" ENVNAME="$env" \
    N_SEEDS=3 EPISODE_NUM=20 N_JOBS=3 LOG_TAG="$tag" \
    bash "$SCRIPT_DIR/eval_model.sh"
}

# 1) 异构联邦全局模型 task3 —— 关机前只跑到 12/20
run_eval "$HETERO_G" mujoco_metaworld_task3 button-press-topdown-v2 hetero_r49
# 2) 异构联邦全局模型 task1 —— 之前只有 1 个 seed，补到 3
run_eval "$HETERO_G" mujoco_metaworld_task1 reach-v2 hetero_r49
# 3) 异构联邦 task1 客户端自己的模型 —— 诊断项，之前只有 1 个 seed
run_eval "$HETERO_C0" mujoco_metaworld_task1 reach-v2 client0_r49
# 4) 等预算集中式 task1（2250 步）—— 之前只有 1 个 seed
run_eval "$EQBUDGET" mujoco_metaworld_task1 reach-v2 equalbudget_task1

echo
echo "===== 门2 最终结果汇总（均值 ± 95% CI）====="
summ() {  # $1=glob  $2=标签
  local hits
  hits=$(ls -1t /root/HPT/output/eval_$1 2>/dev/null | head -1)
  if [ -n "$hits" ]; then
    echo "--- $2  [$hits] ---"
    "$HPT_PY" "$SCRIPT_DIR/summarize_eval.py" "$hits" 2>/dev/null | grep -E 'n=[0-9]' || echo "(解析失败)"
  else
    echo "--- $2: 未找到 eval_$1 ---"
  fi
}
summ "centralized_task1.log"            "集中式 task1 (6000步)"
summ "centralized_task2.log"            "集中式 task2 (6000步)"
summ "centralized_task3.log"            "集中式 task3 (6000步)"
summ "equalbudget_task1*task1*.log"    "等预算集中式 task1 (2250步)"
summ "client0_r49*task1*.log"          "异构 client_0_r49 task1"
summ "hetero_r49*task1*.log"           "异构 global_r49 task1"
summ "hetero_r49*task2*.log"           "异构 global_r49 task2"
summ "hetero_r49*task3*.log"           "异构 global_r49 task3"
summ "iid_r49*task1*.log"              "IID global_r49 task1"
summ "iid_r49*task2*.log"              "IID global_r49 task2"
summ "iid_r49*task3*.log"              "IID global_r49 task3"

echo
echo "===== 全部评估日志 ====="
ls -1t /root/HPT/output/eval_*.log 2>/dev/null | head -30
echo
df -h / | tail -1
echo "FINISH_GATE2_DONE"
