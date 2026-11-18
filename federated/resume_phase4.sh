#!/usr/bin/env bash
# 第4阶段一键续跑（云端入口）。
#
# ============================ 为什么要有这个 ============================
# 2026-11-08 启动的 v2 在跑到一半时实例关机，留下一批 **25 轮制(v1)** 的残留权重。
# 直接续跑会出事：
#   · GlobalStepLRScheduler 的 p = 轮次/NUM_ROUNDS，25 轮制的 r1 已把 p 推到 0.08，
#     而 50 轮制的 r2 的 p 只有 0.04 ⇒ **lr 先冲高再回退**，训练轨迹非单调；
#   · v1 失败在 c_drop50/c_drop75/c_int4，它们的 opt_state_c*.pt 没被清理
#     ⇒ v2 续训时加载了 25 轮制的 **Adam 动量**。
# ⇒ 那些配置的"训练历史"与基线 c_drop00（全程 50 轮制）不同，**曲线不可比**。
#
# 本脚本按顺序做四件事：
#   1. 清洗遗留权重（clean_legacy_rounds.sh，幂等、可用 --dry 先看）
#   2. 跑隐私侧 modality_rev 补跑（参数已修正：--dp-steps 50）
#   3. 跑通信扫描 v2（50 轮 × 7 配置 × 3 路并行）
#   4. 提取两侧结果（privacy + comm，含基线自检）
#
# 幂等：每一步都能单独重跑；已完成的部分按判据跳过。
# 用法：bash resume_phase4.sh [clean|privacy|comm|extract|all]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
HPT_PY=/root/miniconda3/envs/hpt/bin/python
STEP=${1:-all}

log () { echo "[$(date '+%F %T')] $*"; }

# ===== MoE 环境变量（训练与评估必须逐字一致）=====
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half

# ===== code_guard 守护：fedcore 被覆盖时自动恢复 =====
start_guard () {
  if pgrep -f "code_guard.sh" > /dev/null 2>&1; then
    log "code_guard 已在运行"
    return 0
  fi
  if [ -f /root/HPT/output/code_guard.sh ]; then
    nohup setsid bash /root/HPT/output/code_guard.sh > "$OUT/code_guard.log" 2>&1 < /dev/null &
    sleep 2
    log "code_guard 已拉起（pid $(pgrep -f code_guard.sh | head -1)）"
  else
    log "[WARN] 找不到 /root/HPT/output/code_guard.sh，跳过守护"
  fi
}

do_clean () {
  log "===== [1/4] 清洗遗留权重 ====="
  bash "$SCRIPT_DIR/clean_legacy_rounds.sh"
}

do_privacy () {
  log "===== [2/4] 隐私侧 modality_rev 补跑 ====="
  bash "$SCRIPT_DIR/run_privacy_rev.sh"
  log "隐私侧补跑结束 rc=$?"
}

do_comm () {
  log "===== [3/4] 通信扫描 v2（50 轮 × 7 配置 × 3 路并行）====="
  bash "$SCRIPT_DIR/run_comm_scan_v2.sh"
  log "通信扫描结束 rc=$?"
}

do_extract () {
  log "===== [4/4] 提取结果 ====="
  $HPT_PY "$SCRIPT_DIR/extract_comm_results.py" 2>&1 | tail -40
  echo "--- 隐私侧 ---"
  $HPT_PY "$SCRIPT_DIR/extract_privacy_results.py" 2>&1 | tail -20
}

case "$STEP" in
  clean)   start_guard; do_clean ;;
  privacy) start_guard; do_privacy ;;
  comm)    start_guard; do_comm ;;
  extract) do_extract ;;
  all)
    start_guard
    do_clean
    do_privacy
    do_comm
    do_extract
    ;;
  *) echo "用法: bash resume_phase4.sh [clean|privacy|comm|extract|all]"; exit 1 ;;
esac

log "resume_phase4 step=$STEP 结束"
