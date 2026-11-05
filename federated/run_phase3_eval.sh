#!/usr/bin/env bash
# FedVLA 第 3 阶段（MoE + EDA）的【成功率评估】。
#
# 为什么必须有这个脚本
# -------------------
# run_phase3.sh 只负责训练（MoE 补丁 + 4 个实验组），训练完只产出：
#   - 收敛曲线（loss）
#   - EDA 权重报告（eda_report.txt / eda_weights.pt）
# 但论文第 4 章的核心结论是**成功率**，而 extract_phase3_results.py 找的是
#   eval_{tag}_{domain}.log
# 如果没人跑评估，这个脚本跑完 4 个实验组（约 8 小时）后，
# 提取器会得到 4 行 "—"（全空）—— 白跑一整夜。
#
# 评估口径（与门2' 完全一致，不混口径）：
#   3 seeds × 20 rollouts/seed，报均值 ± 95% CI（t 分布）
#   N_JOBS=2 → 2 个 seed 并行，显存 24G 装得下
#
# 变体表（LOG_TAG 必须与 extract_phase3_results.py 的 STAGES 一致）：
#   g2b_central_eq       集中式等预算（门2' E3 产物）   → task1
#   g2b_iid_e4           IID 联邦（门2' E4 产物）        → task1
#   g2b_hetero           异构联邦（门2' E5 产物，基线）  → task1/2/3
#   p3_moe_fedavg        异构 + MoE + FedAvg 等权       → task1/2/3
#   p3_moe_eda           异构 + MoE + EDA 中心性        → task1/2/3
#   p3_moe_eda_cluster   异构 + MoE + EDA 簇感知        → task1/2/3
#   p3_moe_iid           IID  + MoE + FedAvg 等权       → task1
#
# 用法：
#   bash run_phase3_eval.sh                    # 评全部变体
#   STAGES="p3_moe_fedavg" bash run_phase3_eval.sh
#   N_JOBS=1 bash run_phase3_eval.sh           # 单进程（显存紧时）
#
# 幂等：评估日志里已有 >=3 张 "| avg |" 表（= 3 个 seed）就跳过。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/root/HPT/output}"
HPT_PY=/root/miniconda3/envs/hpt/bin/python
STAGES="${STAGES:-"g2b_central_eq g2b_iid_e4 g2b_hetero p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster p3_moe_iid"}"
N_JOBS="${N_JOBS:-2}"
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

# 评估一个变体（按 split 决定任务集）
eval_variant () {
  local tag="$1" split="$2"
  has_stage "$tag" || return 0
  local g; g=$(pick_global "$OUT/$tag")
  if [ -z "$g" ]; then
    log "[SKIP] $tag 无全局模型目录（$OUT/$tag/global_r*）"
    return 0
  fi
  if [ "$split" = "iid" ]; then
    eval_one "$g" mujoco_metaworld_task1 reach-v2 "$tag"
  else
    eval_one "$g" mujoco_metaworld_task1 reach-v2                "$tag"
    eval_one "$g" mujoco_metaworld_task2 push-v2                "$tag"
    eval_one "$g" mujoco_metaworld_task3 button-press-topdown-v2 "$tag"
  fi
}

log "===== 第 3 阶段成功率评估（stages=$STAGES n_jobs=$N_JOBS）====="

# 门2' 的基线（门2' 评估守护已用不同 LOG_TAG 评过一遍，这里用统一 tag 再评一次）
eval_variant g2b_central_eq      iid
eval_variant g2b_iid_e4          iid
eval_variant g2b_hetero          hetero
# 第 3 阶段四个实验组
eval_variant p3_moe_fedavg       hetero
eval_variant p3_moe_eda          hetero
eval_variant p3_moe_eda_cluster  hetero
eval_variant p3_moe_iid          iid

# ── 汇总：重新提取（把 phase3 的评估结果也写进 JSON）────────────────────────
log "===== 提取结果 ====="
"$HPT_PY" "$SCRIPT_DIR/extract_phase3_results.py" --out-root "$OUT" \
  --out "$OUT/phase3_raw.json" 2>&1 | tail -n 40

echo
echo "PHASE3_EVAL_DONE"
