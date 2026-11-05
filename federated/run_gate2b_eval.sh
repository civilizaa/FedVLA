#!/usr/bin/env bash
# 门2' 消融的【成功率评估】：5 个变体，3 seeds × 20 rollouts，口径与集中式基线完全一致。
#
# 背景：run_gate2b.sh 只做训练 + 收敛曲线（loss），
#       但门2' 的核心判据是"E4 能否逼近 E3"，必须看**成功率**而不是 loss。
#       本脚本补齐这一环。
#
# 变体表：
#   V1  IID 联邦 10 轮，只开全局 lr            -> g2b_iid_v1/global_r*
#   V2  IID 联邦 10 轮，全局 lr + Adam 持久化   -> g2b_iid_v2/global_r*
#   E3  集中式等预算 task1（2250 步）          -> g2b_central_eq/task1
#   E4  IID 联邦 50 轮，修复后的联邦基线        -> g2b_iid_e4/global_r*
#   E5  异构联邦 50 轮（第 4 章正式结论）       -> g2b_hetero/global_r*
#
# 用法：
#   bash run_gate2b_eval.sh                 # 评估全部变体
#   STAGES="V1 V2" bash run_gate2b_eval.sh  # 只评指定变体
#   N_JOBS=1 bash run_gate2b_eval.sh        # 单进程（省显存/内存）
#
# 幂等：评估日志里已有 >=3 张 "| avg |" 表（= 3 个 seed）就跳过。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/root/HPT/output}"
HPT_PY=/root/miniconda3/envs/hpt/bin/python
STAGES="${STAGES:-"V1 V2 E3 E4 E5"}"
N_JOBS="${N_JOBS:-3}"
N_SEEDS="${N_SEEDS:-3}"
EPISODE_NUM="${EPISODE_NUM:-20}"
EVAL_LOG_DIR=/root/HPT/output

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

log() { echo "[$(date '+%F %T')] $*"; }

# 最新一轮的全局模型目录
pick_global () {
  ls -dt "$1"/global_r* 2>/dev/null | head -1
}

# 评估是否已经完成（3 个 seed -> 3 张表）
eval_done () {
  local logf="$1" n
  [ -f "$logf" ] || return 1
  n=$(grep -c '| avg' "$logf" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  [ "$n" -ge "$N_SEEDS" ]
}

# eval_one <模型目录> <domain> <envname> <log_tag>
eval_one () {
  local model_dir="$1" domain="$2" envname="$3" log_tag="$4"
  local logf="$EVAL_LOG_DIR/eval_${log_tag}_${domain}.log"

  if [ ! -f "$model_dir/model.pth" ]; then
    log "[SKIP] $log_tag/$domain 缺 model.pth（$model_dir）"
    return 1
  fi
  if eval_done "$logf"; then
    log "[SKIP] $log_tag/$domain 已评估（$(grep -c '| avg' "$logf" 2>/dev/null || true) 张表）"
    return 0
  fi

  log "[EVAL] $log_tag/$domain  env=$envname  seeds=$N_SEEDS episodes=$EPISODE_NUM n_jobs=$N_JOBS"
  MODEL_DIR="$model_dir" DOMAIN="$domain" ENVNAME="$envname" \
    N_SEEDS="$N_SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    LOG_TAG="$log_tag" \
    bash "$SCRIPT_DIR/eval_model.sh" 2>&1 | tail -n 12
  if eval_done "$logf"; then
    log "[OK] $log_tag/$domain 完成"
  else
    log "[!!] $log_tag/$domain 评估未完成（表数=$(grep -c '| avg' "$logf" 2>/dev/null || echo 0)）"
  fi
}

log "===== 门2' 成功率评估（stages=$STAGES n_jobs=$N_JOBS）====="

# ── V1 / V2 / E4：IID 联邦（只有 task1）──────────────────────────────────
run_iid_variant () {
  local stage="$1" tag="$2" root="$3" log_tag="$4"
  has_stage "$stage" || return 0
  local g; g=$(pick_global "$OUT/$root")
  if [ -z "$g" ]; then log "[SKIP] $stage 无全局模型目录（$OUT/$root/global_r*）"; return 0; fi
  eval_one "$g" mujoco_metaworld_task1 reach-v2 "$log_tag"
}
run_iid_variant V1 g2b_iid_v1 "g2b_iid_v1"   g2b_iid_v1_r9
run_iid_variant V2 g2b_iid_v2 "g2b_iid_v2"   g2b_iid_v2_r9
run_iid_variant E4 g2b_iid_e4 "g2b_iid_e4"   g2b_iid_e4_r49

# ── E3：集中式等预算（task1）─────────────────────────────────────────────
if has_stage E3; then
  m="$OUT/g2b_central_eq/task1"
  if [ -d "$m" ]; then
    eval_one "$m" mujoco_metaworld_task1 reach-v2 g2b_central_eq
  else
    log "[SKIP] E3 无集中式等预算目录（$m）"
  fi
fi

# ── E5：异构联邦（三个任务都要）──────────────────────────────────────────
if has_stage E5; then
  g=$(pick_global "$OUT/g2b_hetero")
  if [ -z "$g" ]; then
    log "[SKIP] E5 无全局模型目录（$OUT/g2b_hetero/global_r*）"
  else
    eval_one "$g" mujoco_metaworld_task1 reach-v2                  g2b_hetero_r49
    eval_one "$g" mujoco_metaworld_task2 push-v2                  g2b_hetero_r49
    eval_one "$g" mujoco_metaworld_task3 button-press-topdown-v2   g2b_hetero_r49
  fi
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────
log "===== 门2' 成功率汇总 ====="
"$HPT_PY" "$SCRIPT_DIR/extract_gate2b_results.py" --out-root "$OUT" \
  --out "$OUT/gate2b_results.json" 2>&1 | tail -n 40

echo
echo "GATE2B_EVAL_DONE"
