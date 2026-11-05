#!/usr/bin/env bash
# 第 3 阶段评估守护：等第 3 阶段训练跑完（PHASE3_DONE）后自动做成功率评估。
#
# 为什么需要它
# ------------
# run_phase3.sh 跑完 4 个实验组（异构+MoE 三组 + IID+MoE 一组，约 8 小时）后
# **只产出 loss 曲线和 EDA 权重**，没有成功率。
# 而论文第 4 章的结论（MoE/EDA 是否把异构专有损害 −0.616 压下去）
# 必须靠成功率来判定 —— 没有这一步，一整夜的训练等于白跑。
#
# 时间线
# ------
#   门2' 训练 (GATE2B_DONE)  →  门2' 评估守护接管（GPU 独占）
#   第 3 阶段训练 (PHASE3_DONE) →  本守护接管
# 两个守护各自等自己的标记，天然串行，不抢 GPU。
#
# 用法（云端）：
#   nohup setsid bash /root/HPT/federated/phase3_eval_watch.sh \
#       > /root/HPT/output/phase3_eval_watch.log 2>&1 < /dev/null &
#
# 幂等：评估脚本按「日志里已有 3 张表」跳过已完成的项，重复跑无害。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESUME_LOG=/root/HPT/output/resume_all.log
OUT=/root/HPT/output
N_JOBS="${N_JOBS:-2}"
POLL="${POLL:-60}"

echo "[$(date '+%F %T')] 守护启动，等 PHASE3_DONE（轮询 ${POLL}s）"

deadline=$(( $(date +%s) + ${MAX_WAIT:-86400} ))   # 默认最多等 24 小时
while :; do
  if grep -q "PHASE3_DONE" "$RESUME_LOG" 2>/dev/null; then
    echo "[$(date '+%F %T')] 检测到 PHASE3_DONE，开始评估"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "[$(date '+%F %T')] 等待超时，退出"
    exit 1
  fi
  sleep "$POLL"
done

df -BG /root 2>/dev/null | awk 'NR==2{print "[disk] 可用 " $4 " / " $2}'

N_JOBS="$N_JOBS" bash "$SCRIPT_DIR/run_phase3_eval.sh"

echo "[$(date '+%F %T')] 第 3 阶段评估守护结束"
echo "PHASE3_EVAL_WATCH_DONE"
