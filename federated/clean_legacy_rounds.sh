#!/usr/bin/env bash
# 第4阶段通信扫描：**遗留权重清洗**（一次性，幂等）。
#
# ============================ 为什么必须清 ============================
# 背景：2026-11-08 通信扫描 v1 用 25 轮、v2 改 50 轮。v1 跑崩后残留了一批
#   **25 轮制**的权重目录，v2 直接用 detect_start_round 接着训练 →
#   出现两个硬伤：
#   1) **LR 轨迹非单调**：GlobalStepLRScheduler 的 p = 轮次/NUM_ROUNDS。
#      25 轮制的 r1 已经把 p 推到 0.08；v2 的 50 轮制 r2 的 p 只有 0.04
#      ⇒ lr 先冲高再回退，模型经历了一条非单调的 lr 轨迹。
#   2) **Adam 动量串味**：v1 失败在 c_drop50/c_drop75/c_int4，
#      它们的 opt_state_c*.pt 没被清理 → v2 续训时加载了 25 轮制的 Adam 动量。
#   ⇒ 这些配置的"训练历史"与基线 c_drop00（全程 50 轮制）不同，
#     **丢包/量化曲线不可比** → 论文核心图不能有这种污染。
#
# 判据（逐配置，用**证据**决定，不靠猜）：
#   · scanlog_<tag>.log 里 v2 的启动记录 "从第 N 轮开始"
#       - N == 0        → v2 从 r0 全程 50 轮制 → **保留**
#       - N > 0         → 起点接在 v1 的 25 轮制权重上 → **清空重跑**
#   · 有权重但**没有** v2 启动记录 → 纯 v1 残留 → **清空重跑**
#   · opt_state_c*.pt 的 mtime 早于 v2 启动时刻 → 动量是 v1 的 → **清空重跑**
#
# 用法：bash clean_legacy_rounds.sh [--dry]
#   --dry  只报告不删除
set -u

OUT=${OUT:-/root/HPT/output}
DRY=${1:-}
V2_MARK="$OUT/.r25_logs_backed_up"     # v2 启动时打的标记（用于取 v2 启动时刻）
FAILED=()

log () { echo "[$(date '+%F %T')] $*"; }

# v2 的启动时刻（用于判断 opt_state 是否是 v1 遗留）
V2_START_TS=0
if [ -f "$V2_MARK" ]; then
  V2_START_TS=$(stat -c %Y "$V2_MARK" 2>/dev/null || echo 0)
fi
log "v2 启动时刻戳 = ${V2_START_TS}"

for tag in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  d="$OUT/$tag"
  sl="$OUT/scanlog_${tag}.log"
  [ -d "$d" ] || { log "$tag: 无目录，跳过"; continue; }

  # --- 证据 1：v2 的启动轮次 ---
  start=""
  if [ -f "$sl" ]; then
    start=$(grep -oE '从第 [0-9]+ 轮开始' "$sl" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  fi

  # --- 证据 2：是否还有权重 ---
  nw=$(ls -d "$d"/global_r*/trunk.pth 2>/dev/null | wc -l)
  nopt=$(ls "$d"/opt_state_c*.pt 2>/dev/null | wc -l)

  # --- 证据 3：opt_state 是否早于 v2 启动 ---
  stale_opt=0
  if [ "$nopt" -gt 0 ] && [ "$V2_START_TS" -gt 0 ]; then
    oldest_opt=$(ls -t -r "$d"/opt_state_c*.pt 2>/dev/null | head -1)
    if [ -n "$oldest_opt" ]; then
      ots=$(stat -c %Y "$oldest_opt" 2>/dev/null || echo 0)
      [ "$ots" -lt "$V2_START_TS" ] && stale_opt=1
    fi
  fi

  # --- 判定 ---
  reason=""
  if [ "$nw" -gt 0 ] && [ -z "$start" ]; then
    reason="有 ${nw} 个权重但无 v2 启动记录 ⇒ 纯 v1(25轮制) 残留"
  elif [ -n "$start" ] && [ "$start" != "0" ]; then
    reason="v2 从第 ${start} 轮续跑 ⇒ 起点接在 v1(25轮制) 权重上，lr 轨迹非单调"
  elif [ "$stale_opt" = "1" ]; then
    reason="opt_state_c*.pt 早于 v2 启动 ⇒ Adam 动量是 v1 的"
  fi

  if [ -z "$reason" ]; then
    log "$tag: 保留（start=${start:-无} 权重=${nw} opt=${nopt}）"
    continue
  fi

  if [ "$DRY" = "--dry" ]; then
    log "[DRY] $tag 需清空 —— $reason"
    continue
  fi

  log "[清理] $tag —— $reason"
  # 删权重 / 客户端目录 / Adam 状态（保留日志与 comm_report）
  rm -rf "$d"/global_r* "$d"/client_*_r* 2>/dev/null
  rm -f  "$d"/opt_state_c*.pt 2>/dev/null
  # 训练日志清空（v1/v2 混在一起会误读），旧评估日志改名留档
  if [ -f "$OUT/log_${tag}_train.txt" ]; then
    mv "$OUT/log_${tag}_train.txt" "$OUT/log_${tag}_train.txt.legacy"
  fi
  for dom in mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3; do
    f="$OUT/eval_${tag}_${dom}.log"
    [ -f "$f" ] && mv "$f" "${f}.legacy"
  done
  rm -f "$OUT/scanlog_${tag}.log" "$OUT/.rc_${tag}"
  FAILED+=("$tag")
done

if [ "${#FAILED[@]}" -gt 0 ]; then
  log "已清空并需重跑：${FAILED[*]}"
else
  log "无需清空的配置（全部干净）"
fi
df -h "$OUT" | tail -1 | sed 's/^/[磁盘] /'
