#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段**自动开跑编排器**（2026-11-23）
#
# 流程：
#   1) 等 A/B（评估并行度 N_JOBS=3 vs 1，同一模型同一 domain）跑完
#   2) 比对两份日志里所有 `| avg` 行：
#        完全一致 ⇒ 3 个 seed 并行不改变数值 ⇒ 用 N_JOBS=3（**省 8 小时**）
#        有任何差异 ⇒ 用 N_JOBS=1（保守，与第 3 阶段复用行口径一致）
#   3) 调 phase5_restart.sh（清作废产物 → 起 run_phase5.sh → 验活 → 挂守护）
#
# ⚠️ 为什么要 A/B：第 3 阶段复用的 3 行是 N_JOBS=1 评估的；若并行度会改变数值，
#    主表就混了两种口径 ⇒ 不可比。而"并行不改变数值"必须**实测**，不能假设。
#
# ⚠️ 硬约束：restart.sh 自带三道闸（磁盘 ≥8G / 补丁锚点齐全 / 无残留守护），
#    任一不过即 exit ⇒ 不会带病开跑。
#
# ⚠️ 2026-11-23 修正（第一版 A/B 无效）：
#    第一版在 task1（reach-v2）上做 A/B，而 task1 在异构联邦下**所有变体都是 0.000**
#    ⇒ 两份日志的 avg 行必然都是 0.00 ⇒ 0=0 恒成立 ⇒ **证明不了任何事**。
#    现改用 task3（第 3 阶段 p3_moe_fedavg 在 task3 上 = 0.283，非退化）。
#    并新增「**非退化性检查**」：若两份日志的 avg 全是 0 ⇒ A/B 无分辨力 ⇒ 保守用 N_JOBS=1。
# ==============================================================================
set -u
AB=/root/HPT/output
OUT=/root/HPT/output/phase5
DEC="$AB/ab_decision.txt"
DOMAIN=${DOMAIN:-mujoco_metaworld_task3}

echo "[autostart] 等待 A/B 完成（最多 90 分钟）"
waited=0
while [ "$waited" -lt 90 ]; do
  if grep -q AB_DONE "$AB/ab_status.log" 2>/dev/null; then
    break
  fi
  sleep 60
  waited=$((waited + 1))
done

if ! grep -q AB_DONE "$AB/ab_status.log" 2>/dev/null; then
  echo "[autostart] A/B 超时未完成 ⇒ **不开跑**（避免用未验证的并行度）"
  echo "A/B timeout" > "$DEC"
  exit 1
fi

NJ3="$AB/eval_ab_nj3_${DOMAIN}.log"
NJ1="$AB/eval_ab_nj1_${DOMAIN}.log"
P3REF="$AB/eval_p3_moe_fedavg_${DOMAIN}.log"

# ---- 比对所有 `| avg` 行（去掉多余空格后逐行 diff）----
grep -E '^\|[[:space:]]*avg' "$NJ3" 2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.ab3.txt"
grep -E '^\|[[:space:]]*avg' "$NJ1" 2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.ab1.txt"
grep -E '^\|[[:space:]]*avg' "$P3REF" 2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.abref.txt"

{
  echo "===== 第5阶段 A/B 决策 $(date '+%F %T') ====="
  echo "domain=$DOMAIN"
  echo "--- N_JOBS=3 的 avg 行 ---"; cat "$AB/.ab3.txt"
  echo "--- N_JOBS=1 的 avg 行 ---"; cat "$AB/.ab1.txt"
  echo "--- 第3阶段原始日志的 avg 行（复现性参照）---"; cat "$AB/.abref.txt"
} > "$DEC"

N3=$(wc -l < "$AB/.ab3.txt" 2>/dev/null || echo 0)
N1=$(wc -l < "$AB/.ab1.txt" 2>/dev/null || echo 0)
NREF=$(wc -l < "$AB/.abref.txt" 2>/dev/null || echo 0)

# 🔴 非退化性检查：全 0 ⇒ A/B 没有分辨力（0=0 恒成立）⇒ 不能据此选 N_JOBS=3
DEGEN=0
if [ "$N3" -gt 0 ]; then
  NZ=$(grep -vE '^\|[[:space:]]*avg[[:space:]]*\|[[:space:]]*0\.0+' "$AB/.ab3.txt" | wc -l)
  [ "$NZ" -eq 0 ] && DEGEN=1
fi

NJ=1
if [ "$DEGEN" = "1" ]; then
  echo "[autostart] ⚠️ A/B 的 avg 全是 0 ⇒ **无分辨力**（该 domain 在异构下退化）"
  echo "[autostart] ⇒ 不能据此判定并行度无影响 ⇒ 保守用 N_JOBS=1"
elif [ "$N3" -gt 0 ] && [ "$N3" = "$N1" ] && diff -q "$AB/.ab3.txt" "$AB/.ab1.txt" > /dev/null 2>&1; then
  NJ=3
  echo "[autostart] A/B **完全一致**（$N3 行 avg）⇒ 用 N_JOBS=3（省约 8 小时）"
else
  NJ=1
  echo "[autostart] A/B 不一致或缺失（nj3=$N3 行 / nj1=$N1 行）⇒ 保守用 N_JOBS=1"
fi

# ---- 复现性校验（不参与决策，只留痕）----
if [ "$NREF" -gt 0 ] && [ "$N1" = "$NREF" ] && diff -q "$AB/.ab1.txt" "$AB/.abref.txt" > /dev/null 2>&1; then
  echo "[autostart] ✅ 复现性校验：nj1 与第 3 阶段原始日志逐行一致（$NREF 行）⇒ 主表复用第 3 阶段那 3 行是安全的"
  echo "reproduce_vs_phase3: MATCH" >> "$DEC"
else
  echo "[autostart] ℹ️ 复现性校验：nj1 与第 3 阶段原始日志**不一致或缺参照**（ref=$NREF 行）—— 记录待查"
  echo "reproduce_vs_phase3: MISMATCH_OR_NO_REF" >> "$DEC"
fi

echo "decision: N_JOBS=$NJ" >> "$DEC"

# ---- 开跑 ----
echo "[autostart] 起 phase5_restart.sh（N_JOBS=$NJ）"
NJOBS="$NJ" OUT="$OUT" MAX_HOURS=${MAX_HOURS:-44} POLL=60 PUSH_IDBASE=1010000 \
  bash /root/fedcore/phase5_restart.sh >> "$AB/autostart.log" 2>&1
rc=$?
echo "[autostart] restart rc=$rc" >> "$DEC"
if [ "$rc" -ne 0 ]; then
  echo "[autostart] **开跑失败** ⇒ 见 $AB/autostart.log"
  exit 1
fi
echo "[autostart] 已开跑 ✅（N_JOBS=$NJ）"
