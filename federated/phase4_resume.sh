#!/usr/bin/env bash
# FedVLA 第 4 阶段：通信扫描补跑（第 2 轮）。
#
# 为什么需要这个脚本（2026-11-08）：
#   第 1 轮扫描里 c_drop50 / c_drop75 在 **round 1 聚合就失败**了 ——
#   `run_fedavg.sh` 把丢弃客户端列表传成逗号串 `--drop-clients 0,1`，
#   而 argparse 的 `nargs="*"` 要求空格分隔 ⇒
#   `argument --drop-clients: invalid int value: '0,1'`。
#   修复后需要把这两个配置**重做**。run_comm_scan.sh 本身幂等
#   （按"评估日志齐全"跳过已完成的配置），所以再跑一遍即可。
#
# 本脚本：等第 1 轮跑完 → 自动再跑一轮 → 校验 7 个配置是否评估齐全。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
POLL=${POLL:-60}

log () { echo "[$(date '+%F %T')] $*"; }

log "等待第 1 轮 run_comm_scan.sh 结束…"
waited=0
while pgrep -f "bash /root/HPT/federated/run_comm_scan.sh" > /dev/null; do
  sleep "$POLL"
  waited=$((waited + POLL))
  if [ "$waited" -ge 36000 ]; then
    log "[!!] 等待超时 10 小时，强制开始补跑"
    break
  fi
done
sleep 20
log "第 1 轮已结束，开始补跑第 2 轮（补做 c_drop50 / c_drop75）"

bash "$SCRIPT_DIR/run_comm_scan.sh"
rc=$?
log "第 2 轮结束 rc=$rc"

# ---- 收尾校验：7 个配置的评估日志是否齐全 ----
TAGS="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
MISSING=""
for t in $TAGS; do
  f="$OUT/eval_${t}_mujoco_metaworld_task3.log"
  n=$(grep -c '| avg' "$f" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  if [ "$n" -ge 3 ]; then
    log "[OK] $t 评估齐全（$n 张表）"
  else
    log "[!!] $t 评估不完整（$n 张表）"
    MISSING="$MISSING $t"
  fi
done

if [ -n "$MISSING" ]; then
  log "[!!] 仍未完成的配置：$MISSING"
  exit 1
fi
log "7 个配置评估全部齐全 —— 通信损伤扫描收工"
touch "$OUT/PHASE4_DONE"
log "PHASE4_DONE"
