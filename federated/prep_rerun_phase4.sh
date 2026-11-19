#!/usr/bin/env bash
# ==============================================================================
# 第4阶段「修正版重跑」：**重跑前的清洗**（一次性、幂等、可 --dry 预演）。
#
# ============================ 为什么要专门写这个 ============================
# 2026-11-18 那一轮 50 轮通信扫描跑完了，但两组数字**不能写进论文**：
#
#   (1) c_drop00 / c_drop25 的 scanlog 首行是"从第 **13** 轮开始"
#       ⇒ 它们接在**已作废的 25 轮批次**上续训。
#       GlobalStepLRScheduler 的 p = 轮次/NUM_ROUNDS ⇒ 25 轮制的 p 已到 0.26，
#       50 轮制的 p 只有 0.02 ⇒ **lr 轨迹非单调**，"掉 23.3pp" 其实是
#       「续训 vs 从头训」的差异，不是丢包造成的 ⇒ 丢包对比作废。
#       ⚠️ 2026-11-18 教训："清洗了遗留权重" ≠ "同构" —— 必须**逐个核对
#          7 个 scanlog 的起点**，不能凭"看起来清了"就认为干净。
#
#   (2) c_int8 / c_int4 的成功率**全为 0.000**。
#       离线单测（federated/test_quant_channel.py）证明 int8 量化→聚合只造成
#       **0.21%** 的模型偏差 ⇒ 物理上不可能把成功率从 0.75 打到 0
#       ⇒ 这是**故障**不是信道现象（int4 偏差 3.5%，同样不可能归零）。
#       ⚠️ 铁律：看到"全 0 / 全灭"先算误差量级再下结论。
#
# ⇒ 重跑策略（**只重跑必须重跑的，不浪费算力**）：
#     · 重跑：c_drop00 c_drop25（训练不同构） + c_int8 c_int4 c_bit1（量化口径
#             整组重跑，保证 3 档量化在**同一份修复后代码**下产出，
#             且 bit1 与 int8/int4 同组同口径，可比）
#     · 保留：c_drop50 c_drop75（本来就是 r0 起训、50 轮制、结果有效）
#
#   ⚠️ 为什么 bit1 也重跑：bit1 上次"看起来正常"(0.757)，但它是在**存在量化
#      缺陷的旧代码**下跑出来的。旧代码在 int8/int4 上产出 NaN 级破坏，
#      bit1 只是恰好没崩 ⇒ 不能保证它没受同一缺陷影响。
#      三档量化同组重跑 ⇒ 组内自洽，代价约 +1 路并行（+1~1.5h），可接受。
#
# ============================ 本脚本做什么 ============================
# 对每个"要重跑"的 tag：
#   · 删权重目录 global_r* / client_*_r*  → 保证 detect_start_round 返回 0
#   · 删 Adam 状态 opt_state_c*.pt         → 保证动量不串味
#   · 把旧评估/训练/scanlog/comm_report **改名留档**（.legacy_r2），不删证据
#   · 删 .rc_<tag>                          → 让扫描重新判完成
# 另外清掉：
#   · dp_calib/ 与 .rc_dp_*                 → 强制 C 重新标定
#   · eval_dp_* / dplog_dp_*                → DP 效用扫描从未成功过，清干净
#   · PHASE4_ALL_DONE                       → 旧完成标记作废
#
# 用法：bash prep_rerun_phase4.sh [--dry]
# ==============================================================================
set -u

OUT=${OUT:-/root/HPT/output}

# 要重跑的 5 个配置
RERUN_TAGS=${RERUN_TAGS:-"c_drop00 c_drop25 c_int8 c_int4 c_bit1"}
# 保留不动的 2 个配置（本就是 r0 起训、结果有效）
KEEP_TAGS=${KEEP_TAGS:-"c_drop50 c_drop75"}

DRY=${1:-}
if [ "${DRY:-}" = "--dry" ]; then DRY="1"; fi

log () { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUT" 2>/dev/null || true

log "===== 修正版重跑清洗 ====="
log "要重跑：$RERUN_TAGS"
log "要保留：$KEEP_TAGS"
[ -n "$DRY" ] && log "[DRY] 只报告不删除"

BACKUP_SUFFIX=".legacy_r2"

# ---------------------------------------------------------------------------
# 1) 逐个配置清洗
# ---------------------------------------------------------------------------
for tag in $RERUN_TAGS; do
  d="$OUT/$tag"
  nw=$(ls -d "$d"/global_r*/trunk.pth 2>/dev/null | wc -l)
  nopt=$(ls "$d"/opt_state_c*.pt 2>/dev/null | wc -l)
  start=""
  sl="$OUT/scanlog_${tag}.log"
  if [ -f "$sl" ]; then
    start=$(grep -oE '从第 [0-9]+ 轮开始' "$sl" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  fi
  log "[$tag] 旧起点=${start:-无}  权重=${nw}  opt=${nopt}"

  if [ -n "$DRY" ]; then continue; fi

  # --- 权重 + Adam 状态：直接删（这是"从 r0 起训"的唯一保证）---
  rm -rf "$d"/global_r* "$d"/client_*_r* 2>/dev/null
  rm -f  "$d"/opt_state_c*.pt 2>/dev/null

  # --- 旧结果改名留档（证据不丢，但不会被提取器当成新结果）---
  for f in "$OUT/eval_${tag}"_*.log \
           "$OUT/log_${tag}_train.txt" \
           "$OUT/scanlog_${tag}.log" \
           "$d"/comm_report_r*.json ; do
    [ -e "$f" ] || continue
    mv "$f" "${f}${BACKUP_SUFFIX}" 2>/dev/null
  done
  rm -f "$OUT/.rc_${tag}"
done

# ---------------------------------------------------------------------------
# 2) 保留配置：只做体检（确认它们确实是 r0 起训、评估齐全），**不动文件**
# ---------------------------------------------------------------------------
for tag in $KEEP_TAGS; do
  sl="$OUT/scanlog_${tag}.log"
  start=""
  [ -f "$sl" ] && start=$(grep -oE '从第 [0-9]+ 轮开始' "$sl" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)
  nev=$(grep -c '| avg' "$OUT/eval_${tag}_mujoco_metaworld_task3.log" 2>/dev/null || true)
  [ -n "$nev" ] || nev=0
  log "[$tag] 保留：起点=${start:-无}  task3 评估表数=${nev}"
  if [ "$start" != "0" ]; then
    log "  [!!] $tag 的起点不是 0 —— 与预期不符，需人工确认（本次不自动删）"
  fi
  if [ "$nev" -lt 3 ]; then
    log "  [!!] $tag 的 task3 评估表数 < 3 —— 可能会被扫描脚本判为'未完成'而重训"
  fi
done

# ---------------------------------------------------------------------------
# 3) DP 侧清理（强制重新标定 C + 重新扫描）
# ---------------------------------------------------------------------------
if [ -z "$DRY" ]; then
  rm -rf "$OUT/dp_calib" 2>/dev/null
  for f in "$OUT"/eval_dp_*.log "$OUT"/dplog_dp_*.log "$OUT"/log_dp_*_train.txt ; do
    [ -e "$f" ] && mv "$f" "${f}${BACKUP_SUFFIX}" 2>/dev/null
  done
  rm -f "$OUT"/.rc_dp_*
  rm -f "$OUT/PHASE4_ALL_DONE"
  rm -f "$OUT"/.r25_logs_backed_up.bak
fi

# ---------------------------------------------------------------------------
# 4) 自检：清洗后不该再有这些文件的"活体"
# ---------------------------------------------------------------------------
BAD=0
if [ -z "$DRY" ]; then
for tag in $RERUN_TAGS; do
  for g in "$OUT/$tag"/global_r* "$OUT/$tag"/client_*_r* "$OUT/$tag"/opt_state_c*.pt ; do
    if [ -e "$g" ]; then log "[!!] 残留权重/状态：$g"; BAD=1; fi
  done
  for g in "$OUT/eval_${tag}"_*.log ; do
    # ⚠️ 没有 nullglob 时，未匹配的 glob 会原样展开成 "xxx_*.log" 字面量，
    #    必须先判存在性，否则会**误报残留**（假阳性，会让清洗整体失败）。
    [ -e "$g" ] || continue
    case "$g" in
      *.legacy_r2|*.legacy_r2.*) : ;;
      *) log "[!!] 残留评估日志：$g"; BAD=1 ;;
    esac
  done
  for g in "$OUT/scanlog_${tag}.log" "$OUT/.rc_${tag}" "$OUT/log_${tag}_train.txt" ; do
    if [ -e "$g" ]; then log "[!!] 残留：$g"; BAD=1; fi
  done
done
fi
if [ -n "$DRY" ]; then
  log "[DRY] 跳过自检（未做任何删除）"
elif [ "$BAD" = "0" ]; then
  log "[OK] 自检通过：5 个重跑配置已清空（只剩 .legacy_r2 留档）"
else
  log "[!!] 自检失败：仍有残留，见上面的 [!!] 行"
  exit 1
fi

df -h "$OUT" 2>/dev/null | tail -1 | sed 's/^/[磁盘] /'
log "===== 清洗结束 ====="
