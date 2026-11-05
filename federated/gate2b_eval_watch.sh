#!/usr/bin/env bash
# 门2' 评估守护：等门2' 训练跑完（GATE2B_DONE）后自动做成功率评估。
#
# 为什么需要它：
#   run_gate2b.sh 只做训练 + loss 收敛曲线，
#   但门2' 的核心判据是「E4 能否逼近 E3」，必须看**成功率**。
#   而 resume_all.sh 在门2' 之后会直接进第 3 阶段（MoE+EDA 训练），
#   两边抢 GPU，所以这里单开一条守护链路，训练一结束就接管评估。
#
# 用法（云端）：
#   nohup setsid bash /root/HPT/federated/gate2b_eval_watch.sh \
#       > /root/HPT/output/gate2b_eval_watch.log 2>&1 < /dev/null &
#
# 幂等：评估脚本按「日志里已有 3 张表」跳过已完成的项，重复跑无害。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_LOG=/root/HPT/output/resume_all.log
OUT=/root/HPT/output
N_JOBS="${N_JOBS:-2}"
POLL="${POLL:-60}"

echo "[$(date '+%F %T')] 守护启动，等 GATE2B_DONE（轮询 ${POLL}s）"

deadline=$(( $(date +%s) + ${MAX_WAIT:-43200} ))   # 默认最多等 12 小时
while :; do
  if grep -q "GATE2B_DONE" "$RESUME_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] 检测到 GATE2B_DONE，开始评估"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[$(date '+%F %T')] 等待超时，退出"
    exit 1
  fi
  sleep "$POLL"
done

# 训练刚结束，磁盘可能紧张；顺手看一眼
df -BG /root 2>/dev/null | awk 'NR==2{print "[disk] 可用 " $4 " / " $2}'

N_JOBS="$N_JOBS" bash "$SCRIPT_DIR/run_gate2b_eval.sh"

echo "[$(date '+%F %T')] 评估守护结束"
echo "GATE2B_EVAL_WATCH_DONE"
