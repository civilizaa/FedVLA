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

# ===== MoE 评估环境变量（2026-11-06 修复）====================================
# ⚠️ 这是一个**致命缺陷**的修复，必须读：
#   - 训练侧 run_fedavg.sh 有 `export FEDVLA_MOE=1`，
#     所以 policy 里会创建 8 个专家的 MoEBlock（block 8~15）。
#   - 评估侧原来【一个都没有设】→ `patch_hpt_moe.py` 的判定是
#     `if os.environ.get("FEDVLA_MOE","0") == "1"`，否则 moe_cfg=None
#     → 评估时 policy **根本不创建 MoE 模块**。
#   - 而 `Policy.load_model` 用的是 `load_state_dict(..., strict=False)`
#     → 权重里 264 个 MoE 键被**静默丢弃**，不报错、不告警。
#   实测（probe_moe_eval.py，global_r44）：
#     不设 FEDVLA_MOE → policy 12.644M 参数，unexpected_moe=264（残缺模型）
#     设   FEDVLA_MOE → policy 21.082M 参数，missing/unexpected_moe=0（完整模型）
#   ⇒ 不修这一条，第 3 阶段 10 次评估全部评的是「没有 MoE 的模型」，
#     4 个实验组的成功率全部作废，且没有任何报错线索。
#
# 修法：**不能改 eval_model.sh**（评估时它正在被 bash 执行，
# 改动会让 bash 读到错乱字节导致评估崩在半路），
# 改为在【本脚本】里按变体 export，子 shell 继承即可。
#
# 参数值必须与训练侧 run_phase3.sh 的默认值逐字一致
# （num_experts / top_k / expert_ratio / layers），否则结构对不上。
MOE_EXPERTS="${MOE_EXPERTS:-8}"
MOE_TOPK="${MOE_TOPK:-2}"
MOE_EXPERT_RATIO="${MOE_EXPERT_RATIO:-0.25}"
MOE_LAYERS="${MOE_LAYERS:-half}"

apply_moe_env () {
  if [ "$1" = "1" ]; then
    export FEDVLA_MOE=1
    export FEDVLA_MOE_EXPERTS="$MOE_EXPERTS"
    export FEDVLA_MOE_TOPK="$MOE_TOPK"
    export FEDVLA_MOE_EXPERT_RATIO="$MOE_EXPERT_RATIO"
    export FEDVLA_MOE_LAYERS="$MOE_LAYERS"
  else
    unset FEDVLA_MOE FEDVLA_MOE_EXPERTS FEDVLA_MOE_TOPK \
          FEDVLA_MOE_EXPERT_RATIO FEDVLA_MOE_LAYERS
  fi
}

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
log() { echo "[$(date '+%F %T')] $*"; }

# 最新一轮的全局模型目录
pick_global () {
  ls -dt "$1"/global_r* 2>/dev/null | head -1
}

# 模型的**实际目录**。两种布局：
#   联邦变体：$OUT/<name>/global_r<N>/          （pick_global）
#   集中式变体：$OUT/<name>/task1/              （没有 global_r*！）
# ⚠️ 2026-11-05 修正：g2b_central_eq 下只有 task1/，用 pick_global 会返回空
#    → E3 基线被静默跳过，表格里 E3 列全空。
pick_model_dir () {
  local root="$1"
  local g; g=$(pick_global "$root")
  if [ -n "$g" ]; then echo "$g"; return 0; fi
  for cand in "$root"/task1 "$root"/task2 "$root"/task3; do
    [ -f "$cand/model.pth" ] && { echo "$cand"; return 0; }
  done
  echo ""
}

# 评估是否已经完成（3 个 seed -> 3 张表）
eval_done () {
  local logf="$1" n
  [ -f "$logf" ] || return 1
  n=$(grep -c '| avg' "$logf" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  [ "$n" -ge "$N_SEEDS" ]
}

# 表格数（用于日志输出）。⚠️ 不能写 `grep -c ... || echo 0`：
#    grep 无匹配时**既输出 0 又退出码 1**，`|| echo 0` 会追加第二行 "0"
#    → 变量变成 "0\n0" → `[ "$n" -ge 3 ]` 报 "integer expression expected"
#    → 评估被**静默跳过**（历史踩坑）。
table_count () {
  local n
  n=$(grep -c '| avg' "$1" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  echo "$n"
}

# 抢「某个日志文件正在被评估」的锁（原子 mkdir）。
# 目录里写 PID，便于识别进程已死的**残留锁**。
acquire_eval_lock () {
  local lockdir="$1"
  if mkdir "$lockdir" 2>/dev/null; then
    echo $$ > "$lockdir/pid"
    return 0
  fi
  # 锁存在 → 检查是不是残留锁（进程已死）
  local oldpid
  oldpid=$(cat "$lockdir/pid" 2>/dev/null || echo "")
  if [ -n "$oldpid" ] && ! kill -0 "$oldpid" 2>/dev/null; then
    log "[WARN] 清理残留评估锁 $lockdir（pid=$oldpid 已死）"
    rm -rf "$lockdir" 2>/dev/null || true
    if mkdir "$lockdir" 2>/dev/null; then
      echo $$ > "$lockdir/pid"
      return 0
    fi
  fi
  return 1
}

release_eval_lock () { rm -rf "$1" 2>/dev/null || true; }

# 等另一个评估进程把这一项跑完（拿不到锁时调用）。
# 返回 0 = 这一项已被别人评完（可跳过）；1 = 仍然没评完。
wait_for_eval_lock () {
  local lockdir="$1" logf="$2" waited=0
  while [ "$waited" -lt "${EVAL_LOCK_WAIT:-5400}" ]; do
    sleep 30
    waited=$((waited + 30))
    if ! [ -d "$lockdir" ]; then
      # 锁释放了 → 看结果有没有写出来
      if eval_done "$logf"; then
        return 0
      fi
      # 锁没了但结果没出来 → 对方可能失败了，重新自己来
      return 1
    fi
  done
  log "[WARN] 等待评估锁 $lockdir 超时（${waited}s）"
  return 1
}

# eval_one <模型目录> <domain> <envname> <log_tag> <moe: 0|1>
eval_one () {
  local model_dir="$1" domain="$2" envname="$3" log_tag="$4" moe="${5:-0}"
  local logf="$EVAL_LOG_DIR/eval_${log_tag}_${domain}.log"
  local lockdir="$EVAL_LOG_DIR/.lock_eval_${log_tag}_${domain}"

  if [ ! -f "$model_dir/model.pth" ]; then
    log "[SKIP] $log_tag/$domain 缺 model.pth（$model_dir）"
    return 1
  fi
  if eval_done "$logf"; then
    log "[SKIP] $log_tag/$domain 已评估（$(table_count "$logf") 张表）"
    return 0
  fi

  # ── 抢锁：防止两个守护（提前评估 / 正式评估）同时评同一个模型 ──────────
  if ! acquire_eval_lock "$lockdir"; then
    log "[WAIT] $log_tag/$domain 已有评估进程在跑，等待其完成..."
    if wait_for_eval_lock "$lockdir" "$logf"; then
      log "[SKIP] $log_tag/$domain 已由其他进程评估完成（$(table_count "$logf") 张表）"
      return 0
    fi
    # 对方失败/超时 → 清理残留锁，自己再试一次
    release_eval_lock "$lockdir"
    if ! acquire_eval_lock "$lockdir"; then
      log "[!!] $log_tag/$domain 抢不到评估锁，跳过本项（需人工核查）"
      return 1
    fi
  fi

  # 关键：按变体开关 MoE 环境变量。子 shell（eval_model.sh）会继承。
  apply_moe_env "$moe"

  log "[EVAL] $log_tag/$domain  env=$envname  seeds=$N_SEEDS episodes=$EPISODE_NUM n_jobs=$N_JOBS  moe=$moe"
  MODEL_DIR="$model_dir" DOMAIN="$domain" ENVNAME="$envname" \
    N_SEEDS="$N_SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    LOG_TAG="$log_tag" \
    bash "$SCRIPT_DIR/eval_model.sh" 2>&1 | tail -n 12
  release_eval_lock "$lockdir"

  # ===== 自验证：MoE 变体必须看到 ~22.4M 参数，否则说明 MoE 没生效 =====
  # run_eval.py 会打印 "number of params (M): X"。
  #   实测（P3-A global_r49，走完整 hydra 路径）：
  #     不设 FEDVLA_MOE → 13.94M（残缺模型，264 个专家权重被 strict=False 静默丢弃）
  #     设   FEDVLA_MOE → 22.38M（完整模型，+8.44M = 8 专家 × 8 层的残差分支）
  #   阈值取 18.0M：落在 13.94 与 22.38 之间，两侧余量都够。
  # 少了这一步，"评估了残缺模型"会再次静默发生（这正是本次事故的成因）。
  if [ "$moe" = "1" ]; then
    local npm; npm=$(grep -oE "number of params \(M\): [0-9.]+" "$logf" 2>/dev/null | head -1 | grep -oE "[0-9.]+$" || true)
    if [ -z "$npm" ]; then
      log "[!!] $log_tag/$domain 日志里找不到参数量 —— 无法确认 MoE 是否生效，人工核查"
    else
      local ok=0
      awk -v v="$npm" 'BEGIN{ if (v > 18.0) exit 0; else exit 1 }' && ok=1
      if [ "$ok" = "1" ]; then
        log "[OK] $log_tag/$domain 参数量 ${npm}M（>18M，MoE 已生效）"
      else
        log "[!!] $log_tag/$domain 参数量仅 ${npm}M —— **MoE 没生效，评的是残缺模型**，本项结果作废"
      fi
    fi
  fi

  if eval_done "$logf"; then
    log "[OK] $log_tag/$domain 完成"
  else
    log "[!!] $log_tag/$domain 评估未完成（表数=$(table_count "$logf")）"
  fi
}

# eval_variant <目录名> <split> [log_tag]
#   MoE 开关来自环境变量 MOE_ON（0|1），**不用位置参数**：
#   位置参数一旦多传/少传一位，log_tag 就会被错位成 "0"，
#   日志名变成 eval_0_xxx → eval_done 判不出来 → 重复评估 / 结果错位。
eval_variant () {
  local tag="$1" split="$2" log_tag="${3:-$1}"
  local moe="${MOE_ON:-0}"
  has_stage "$tag" || return 0
  local g; g=$(pick_model_dir "$OUT/$tag")
  if [ -z "$g" ]; then
    log "[SKIP] $tag 无可用模型目录（$OUT/$tag/global_r* 或 task1/）"
    return 0
  fi
  if [ "$split" = "iid" ]; then
    eval_one "$g" mujoco_metaworld_task1 reach-v2 "$log_tag" "$moe"
  else
    eval_one "$g" mujoco_metaworld_task1 reach-v2                "$log_tag" "$moe"
    eval_one "$g" mujoco_metaworld_task2 push-v2                "$log_tag" "$moe"
    eval_one "$g" mujoco_metaworld_task3 button-press-topdown-v2 "$log_tag" "$moe"
  fi
}

log "===== 第 3 阶段成功率评估（stages=$STAGES n_jobs=$N_JOBS）====="

# 门2' 的基线（门2' 评估守护已用不同 LOG_TAG 评过一遍，这里用统一 tag 再评一次）
# ⚠️ 基线是**无 MoE** 的模型 → MOE_ON=0
MOE_ON=0
eval_variant g2b_central_eq      iid    g2b_central_eq
eval_variant g2b_iid_e4          iid    g2b_iid_e4_r49
eval_variant g2b_hetero          hetero g2b_hetero_r49
# 第 3 阶段四个实验组（**全部含 MoE** → MOE_ON=1）
MOE_ON=1
eval_variant p3_moe_fedavg       hetero
eval_variant p3_moe_eda          hetero
eval_variant p3_moe_eda_cluster  hetero
eval_variant p3_moe_iid          iid
MOE_ON=0

# ── 汇总：重新提取（把 phase3 的评估结果也写进 JSON）────────────────────────
log "===== 提取结果 ====="
"$HPT_PY" "$SCRIPT_DIR/extract_phase3_results.py" --out-root "$OUT" \
  --out "$OUT/phase3_raw.json" 2>&1 | tail -n 40

echo
echo "PHASE3_EVAL_DONE"
