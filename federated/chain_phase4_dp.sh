#!/usr/bin/env bash
# FedVLA 第4阶段：**串行接力链**（通信扫描 → DP 效用扫描）。
#
# 为什么要串行而不是并跑：
#   通信扫描是 3 路并行、占满 GPU（3×5.4G）+ 评估的 MuJoCo 显存；
#   再挤 DP 实验有 OOM 风险 —— OOM 会让**正在跑的通信扫描**崩掉，
#   那是 9 小时的算力。所以 DP 实验必须等通信扫描彻底结束后再开始。
#
# 完成判据（不用权重目录）：
#   通信扫描的每个配置结束时写 $OUT/.rc_<tag>，7 个 .rc 全部出现
#   ⇒ 7 个配置的「训练 + 评估」都已结束（含失败）。
#   权重目录会被 CLEAN_AFTER_EVAL 删掉，不能当判据（历史踩坑：判据失效 →
#   已完成配置被重训 7 小时）。
#
# 死锁保护：若通信扫描进程已不存在且 40 分钟内没有新的 .rc 出现，
#   判定通信扫描已中断（实例被关机/脚本被杀），**不无限等待**，直接进入 DP 阶段。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
POLL=${POLL:-120}                # 轮询间隔（秒）
TIMEOUT_S=${TIMEOUT_S:-57600}    # 最长等待通信扫描的秒数（16 小时）

COMM_TAGS="c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1"
EXPECTED=7

log () { echo "[$(date '+%F %T')] $*"; }

count_rc () {
  local n=0 t
  for t in $COMM_TAGS; do [ -f "$OUT/.rc_${t}" ] && n=$((n + 1)); done
  echo "$n"
}

comm_proc_alive () {
  pgrep -f "run_comm_scan_v2" > /dev/null 2>&1 && return 0
  pgrep -f "run_fedavg.sh" > /dev/null 2>&1 && return 0
  return 1
}

log "===== 第4阶段串行接力链启动 ====="
log "等待通信扫描完成（判据：${EXPECTED} 个 .rc_<tag> 全部出现）"

waited=0
last_n=0
stall=0
while : ; do
  n=$(count_rc)
  if [ "$n" -ge "$EXPECTED" ]; then
    log "[OK] 通信扫描完成（$n/${EXPECTED}）"
    break
  fi
  if [ "$n" -gt "$last_n" ]; then
    stall=0
    log "通信扫描进度：$n/${EXPECTED} 个配置已结束"
  else
    stall=$((stall + 1))
  fi
  last_n=$n

  # 死锁保护：脚本进程已死 + 长时间无进展 ⇒ 不再等
  if ! comm_proc_alive && [ "$stall" -ge 20 ]; then
    log "[!!] 通信扫描进程已不存在且 40 分钟无新进展（$n/${EXPECTED}）—— 判定已中断"
    log "     继续进入 DP 阶段（DP 实验不依赖通信结果，基线会标注缺失）"
    break
  fi
  if [ "$waited" -ge "$TIMEOUT_S" ]; then
    log "[!!] 等待通信扫描超时（${TIMEOUT_S}s，$n/${EXPECTED}）—— 继续进入 DP 阶段"
    break
  fi
  sleep "$POLL"
  waited=$((waited + POLL))
done

# ===== 阶段 1：C 标定 =====
log "===== 阶段 1：DP 效用扫描的 C 标定 ====="
bash "$SCRIPT_DIR/calibrate_dp_clip.sh"
rc=$?
if [ "$rc" != "0" ]; then
  log "[!!] 标定失败（rc=$rc）—— DP 效用扫描无法进行，链路终止"
  exit 1
fi

# ===== 阶段 2：DP 效用扫描 =====
log "===== 阶段 2：DP 噪声 vs 真实任务成功率 ====="
bash "$SCRIPT_DIR/run_dp_utility.sh"
rc=$?
if [ "$rc" != "0" ]; then
  log "[!!] DP 效用扫描未全部完成（rc=$rc）—— 可重跑本脚本续跑（幂等）"
  exit 1
fi

# ===== 阶段 3：提取结果 =====
log "===== 阶段 3：提取 DP 效用结果 ====="
/root/miniconda3/envs/hpt/bin/python "$SCRIPT_DIR/extract_dp_utility.py" \
  --out "$OUT/dp_utility_raw.json"
rc=$?
if [ "$rc" != "0" ]; then
  log "[!!] 提取失败（rc=$rc）"
  exit 1
fi

log "===== 第4阶段接力链全部完成 ====="
echo "PHASE4_CHAIN_DONE"
