#!/usr/bin/env bash
# FedVLA 第 4 阶段守护：通信损伤扫描。
#
# 为什么要有这个守护（而不是直接在 resume_all.sh 里同步跑）：
#   relay agent 的单命令上限 MAX_CMD_WAIT=600s，超时会 **SIGKILL 整个进程组**。
#   第4阶段 7 个配置 × 25 轮 ≈ 6 小时，远超上限 → 必须 nohup setsid 起、
#   本地轮询日志文件，绝不能挂 agent 前台。
#
# 为什么先等第3阶段评估收工：
#   两个阶段同时评估会打爆 15 核（3 个 joblib worker × OMP_NUM_THREADS=8 = 24 线程），
#   实测 load average 冲到 22~29，评估慢一倍不止。第3阶段只剩 P3-D 一项，
#   等它跑完再开第4阶段，省时间也省心。
#
# 幂等：run_comm_scan.sh 自身幂等（已完成的配置按"训练到最后一轮 + 评估齐全"跳过）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
WAIT_TIMEOUT=${WAIT_TIMEOUT:-7200}     # 等第3阶段评估的最长秒数（超时则照常开跑）
POLL=${POLL:-120}

log () { echo "[$(date '+%F %T')] $*"; }

# 第3阶段评估是否全部完成（10 个日志，每个 >= 3 张表）
phase3_eval_ready () {
  local f n cnt=0
  shopt -s nullglob
  for f in "$OUT"/eval_p3_moe_*.log; do
    n=$(grep -c '| avg' "$f" 2>/dev/null || true)
    [ -n "$n" ] || n=0
    cnt=$((cnt + 1))
    [ "$n" -ge 3 ] || return 1
  done
  shopt -u nullglob
  [ "$cnt" -ge 10 ] || return 1
  return 0
}

log "第4阶段守护启动：等第3阶段评估收工（最多 ${WAIT_TIMEOUT}s）"
waited=0
while :; do
  if phase3_eval_ready; then
    log "第3阶段评估已全部完成（10 项 × 3 seeds）"
    break
  fi
  if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then
    log "[WARN] 等第3阶段评估超时 ${WAIT_TIMEOUT}s —— 仍然开始第4阶段"
    break
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

log "开始第4阶段：通信损伤扫描"
bash "$SCRIPT_DIR/run_comm_scan.sh"
rc=$?
log "第4阶段结束 rc=$rc"

if [ "$rc" = "0" ]; then
  touch "$OUT/PHASE4_DONE"
  log "PHASE4_DONE"
else
  log "[!!] 第4阶段未正常结束（rc=$rc）—— 重新运行本脚本可续跑（幂等）"
fi
