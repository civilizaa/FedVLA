#!/usr/bin/env bash
# ==============================================================================
# 第 5 阶段**自动开跑编排器**（2026-11-23，第三版）
#
# 流程：
#   1) 等 A/B（同一模型同一 domain，N_JOBS=3 vs 1）跑完
#   2) 🔴 判决规则（v3 修正，v2 错）：**主表要用「能复现第 3 阶段原始口径」的并行度**
#      —— 主表 balanced 臂有 3 行是**复用第 3 阶段**的评估结果；
#        若新跑的行用另一种并行度评估 ⇒ 同一张表混两种口径 ⇒ 与复用行不可比（且不报错）。
#      ⇒ 判据 = 「nj3 的 avg 行」/「nj1 的 avg 行」哪一组与**第 3 阶段原始日志**逐行相同。
#        （⚠️ 不能用「nj3 vs nj1 是否相同」来判：v2 就是这么写的，结论反了。）
#   3) 调 phase5_restart.sh（清作废产物 → 起 run_phase5.sh → 验活 → 挂守护）
#
# ⚠️ 为什么必须实测：评估的 GPU 并发会改变浮点归约顺序 ⇒ 动作微小偏移 ⇒
#    MuJoCo 混沌动力学放大 ⇒ 个别 episode 翻转。实测 seed0：并行 0.55 vs 串行 0.50。
#    所以"并行只省时间不改数值"是**错的假设**，必须实测。
#
# ⚠️ 硬约束：restart.sh 自带三道闸（磁盘 ≥8G / 补丁锚点齐全 / 无残留守护），
#    任一不过即 exit ⇒ 不会带病开跑。
# ==============================================================================
set -u
AB=/root/HPT/output
OUT=/root/HPT/output/phase5
DEC="$AB/ab_decision.txt"
DOMAIN=${DOMAIN:-mujoco_metaworld_task3}
REF=${REF:-eval_p3_moe_fedavg_${DOMAIN}.log}

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
P3REF="$AB/$REF"

grep -E '^\|[[:space:]]*avg' "$NJ3"  2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.ab3.txt"
grep -E '^\|[[:space:]]*avg' "$NJ1"  2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.ab1.txt"
grep -E '^\|[[:space:]]*avg' "$P3REF" 2>/dev/null | sed 's/[[:space:]]\+/ /g' > "$AB/.abref.txt"

{
  echo "===== 第5阶段 A/B 决策 $(date '+%F %T') ====="
  echo "domain=$DOMAIN   reference=$REF"
  echo "--- N_JOBS=3 的 avg 行 ---"; cat "$AB/.ab3.txt"
  echo "--- N_JOBS=1 的 avg 行 ---"; cat "$AB/.ab1.txt"
  echo "--- 第3阶段原始日志的 avg 行（复用行的口径基准）---"; cat "$AB/.abref.txt"
} > "$DEC"

N3=$(wc -l < "$AB/.ab3.txt" 2>/dev/null || echo 0)
N1=$(wc -l < "$AB/.ab1.txt" 2>/dev/null || echo 0)
NREF=$(wc -l < "$AB/.abref.txt" 2>/dev/null || echo 0)

same () { diff -q "$1" "$2" > /dev/null 2>&1; }

M3=0; M1=0
[ "$N3" -gt 0 ] && [ "$N3" = "$NREF" ] && same "$AB/.ab3.txt" "$AB/.abref.txt" && M3=1
[ "$N1" -gt 0 ] && [ "$N1" = "$NREF" ] && same "$AB/.ab1.txt" "$AB/.abref.txt" && M1=1

# 🔴 非退化性检查：基准日志全 0 ⇒ 比对毫无分辨力（0=0 恒成立）⇒ 不能据此判决
DEGEN=0
if [ "$NREF" -gt 0 ]; then
  NZ=$(grep -vE '^\|[[:space:]]*avg[[:space:]]*\|[[:space:]]*0\.0+' "$AB/.abref.txt" | wc -l)
  [ "$NZ" -eq 0 ] && DEGEN=1
fi

NJ=1
if [ "$DEGEN" = "1" ]; then
  echo "[autostart] ⚠️ 基准日志的 avg 全是 0 ⇒ **比对无分辨力** ⇒ 保守用 N_JOBS=1"
  echo "decision_basis: DEGENERATE_REFERENCE" >> "$DEC"
elif [ "$M3" = "1" ] && [ "$M1" = "1" ]; then
  NJ=3
  echo "[autostart] nj3 与 nj1 **都与第3阶段原始口径相同** ⇒ 评估确定性 ⇒ 用 N_JOBS=3"
  echo "decision_basis: BOTH_MATCH_REFERENCE" >> "$DEC"
elif [ "$M3" = "1" ]; then
  NJ=3
  echo "[autostart] ✅ nj3 **复现了第3阶段原始口径**（nj1 没有）⇒ 用 N_JOBS=3（复用行与新跑行口径一致）"
  echo "decision_basis: NJ3_MATCHES_REFERENCE" >> "$DEC"
elif [ "$M1" = "1" ]; then
  NJ=1
  echo "[autostart] ✅ 只有 nj1 复现了第3阶段原始口径 ⇒ 用 N_JOBS=1"
  echo "decision_basis: NJ1_MATCHES_REFERENCE" >> "$DEC"
else
  NJ=3
  echo "[autostart] ⚠️ nj3 与 nj1 **都没复现**第3阶段原始口径 ⇒ 复用行口径存疑"
  echo "[autostart] ⇒ 选 N_JOBS=3（最快），**收尾时必须复核 3 行复用数据的口径**"
  echo "decision_basis: NEITHER_MATCHES_REFERENCE_REVIEW_NEEDED" >> "$DEC"
fi

echo "decision: N_JOBS=$NJ" >> "$DEC"
echo "reproduce_vs_phase3: nj3_match=$M3 nj1_match=$M1" >> "$DEC"

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
