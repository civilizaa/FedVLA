#!/usr/bin/env bash
# ==============================================================================
# FedVLA 第 5 阶段：主实验矩阵 + 消融（论文路线图 6.1 / 6.2）
#
# 目的
# ----
# 论文的主表需要 6 个方法 × 2 种数据划分的对比。本脚本把矩阵**参数化**并
# 串行执行，每个配置独立目录、独立 .rc 判据，可断点续跑。
#
# 主实验矩阵（路线图 6.1）
# ----------------------
#   方法            均衡划分(balanced)      含孤立客户端(isolated)
#   Centralized     ✅ 复用第3阶段 g2b_central_eq      —
#   FedAvg          ✅ 复用第3阶段 p3_moe_fedavg      🔄 本脚本
#   FedProx         🔄 本脚本                         🔄 本脚本
#   SCAFFOLD        🔄 本脚本                         🔄 本脚本
#   EDA（复现）      ✅ 复用第3阶段 p3_moe_eda         🔄 本脚本
#   按簇聚合（本文）  ✅ 复用第3阶段 p3_moe_eda_cluster 🔄 本脚本
#
# ⚠️ "复用"的前提是**训练历史同构**（铁律 D）：
#    均衡划分的定义 = 第 3 阶段的默认异构划分（task1/2/3/1，每客户端 100 条示范，
#    50 轮 × 3 本地 epoch，USE_MOE=1 + 全局 lr + Adam 持久化）⇒ 逐字同构，可复用。
#    本脚本的 isolated 臂与之**不可比**（划分不同）⇒ 必须重跑。
#
# 两种划分的定义（HPT 约束：每个客户端只能挂 1 个任务，因为 hpt.run 只用
# domains 列表的第一个域构建数据集与 stem/head）
# -------------------
#   balanced（均衡划分）：
#       client0=t1(reach)  client1=t2(push)  client2=t3(button)  client3=t1(reach)
#       每客户端 EPISODE_CNT=100 条示范 ⇒ **数据量均衡**，任务异构。
#       ⚠️ epoch_size 不同（15/19/24/15）是因为任务 episode 长度不同，
#          不是数据量不同。
#
#   isolated（含孤立客户端）：
#       client0=t1 独占（**没有任何其它客户端有 t1** ⇒ 天然的孤岛客户端）
#       client1=t2  client2=t3  client3=t2   （t2 被两个客户端共享）
#       ⇒ 检验"孤岛客户端"场景下：① 联邦方法是否被拖垮；② 簇感知聚合能否
#          把孤岛客户端单独聚成一簇（H2 的核心卖点）。
#
# 用法（云端）
# ----------
#   bash /root/fedcore/run_phase5.sh                 # 跑全部
#   STAGES="P5-A" bash /root/fedcore/run_phase5.sh   # 只跑某一段
#
# 可调环境变量
#   OUT          输出根目录（默认 /root/HPT/output/phase5）
#   NUM_ROUNDS   联邦轮数（默认 50，必须与第 3 阶段一致）
#   SEEDS        评估 seed 数（默认 3 —— 论文统计底线）
#   EPISODE_NUM  每 seed 的 rollout 数（默认 20）
#   FORCE=1      忽略已有产物，全部重跑（默认 0）
#   STAGES       要跑的段（默认 "P5-A P5-B"）
# ==============================================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT=${OUT:-/root/HPT/output/phase5}
LOGDIR="$OUT/logs"
NUM_ROUNDS=${NUM_ROUNDS:-50}
SEEDS=${SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-1}
FORCE=${FORCE:-0}
STAGES=${STAGES:-"P5-A P5-B"}
HPT_PY=${HPT_PY:-/root/miniconda3/envs/hpt/bin/python}

# MoE 配置（必须与第 3 阶段逐字一致，否则不可比）
MOE_EXPERTS=${MOE_EXPERTS:-8}
MOE_TOPK=${MOE_TOPK:-2}
MOE_EXPERT_RATIO=${MOE_EXPERT_RATIO:-0.25}
MOE_LAYERS=${MOE_LAYERS:-half}

mkdir -p "$OUT" "$LOGDIR"

# 本阶段涉及的全部配置 tag（preflight 用它判断是否需要 HPT 基线补丁）
ALL_TAGS="p5_bal_fedprox p5_bal_scaffold p5_iso_fedavg p5_iso_fedprox p5_iso_scaffold p5_iso_eda p5_iso_edaclu"

log ()  { echo "[$(date '+%F %T')] $*"; }
banner () { echo; echo "=============================================================="; echo ">>> $*"; echo "=============================================================="; }

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# ==============================================================================
# 开训前硬拦截（fail-loud，铁律 Q）
# ==============================================================================
preflight () {
  local ok=1
  if [ ! -f "$SCRIPT_DIR/fed_baselines.py" ]; then
    log "[!!] 缺 $SCRIPT_DIR/fed_baselines.py —— 第 5 阶段基线代码未同步"; ok=0
  fi
  if [ ! -f "$SCRIPT_DIR/fedvla_baseline_hook.py" ]; then
    log "[!!] 缺 $SCRIPT_DIR/fedvla_baseline_hook.py —— 钩子源文件未同步"; ok=0
  fi
  # 需要基线的配置 ⇒ HPT 必须已打补丁
  if grep -q "fedprox\|scaffold" <<< "$ALL_TAGS"; then
    if ! grep -q "apply_baseline_hook" /root/HPT/hpt/train_test.py 2>/dev/null; then
      log "[!!] HPT 未打联邦基线补丁（hpt/train_test.py 无 apply_baseline_hook）"
      log "    先执行： python $SCRIPT_DIR/patch_hpt_fedbaseline.py --hpt-dir /root/HPT"
      ok=0
    else
      log "[OK] HPT 联邦基线补丁已就位"
    fi
  fi
  # HPT 必须可 import（铁律 F/G：换个调用方就炸 ⇒ 先硬拦截）
  if ! ( cd /root/HPT && "$HPT_PY" -c "import hpt" 2>/dev/null ); then
    log "[!!] HPT 不可 import（cd /root/HPT 后仍失败）"; ok=0
  fi
  [ "$ok" = "1" ] || { log "=== 预检失败，未开跑 ==="; exit 1; }
  log "[OK] 预检通过"
}

# ==============================================================================
# 划分定义
# ==============================================================================
BAL_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1"
BAL_ENVS="reach-v2 push-v2 button-press-topdown-v2 reach-v2"
BAL_ESIZE="15 19 24 15"

# 孤立客户端：client0 独占 t1；其余共享 t2/t3（HPT 一客户端一域 ⇒ 逐个展开）
ISO_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task2"
ISO_ENVS="reach-v2 push-v2 button-press-topdown-v2 push-v2"
ISO_ESIZE="15 19 24 19"

# ==============================================================================
# 配置矩阵：tag | 划分 | 聚合模式 | 基线 | 任务列表
# ==============================================================================
CFG_TAGS=()
CFG_SPLIT=()
CFG_AGG=()
CFG_BASE=()
add_cfg () { CFG_TAGS+=("$1"); CFG_SPLIT+=("$2"); CFG_AGG+=("$3"); CFG_BASE+=("$4"); }

if has_stage P5-A; then
  # 均衡划分：FedAvg / EDA / 簇感知 复用第 3 阶段 ⇒ 这里只跑两个新基线
  add_cfg p5_bal_fedprox  balanced fedavg       fedprox
  add_cfg p5_bal_scaffold balanced fedavg       scaffold
fi
if has_stage P5-B; then
  # 孤立客户端：全部方法重跑（划分与第 3 阶段不同 ⇒ 不可复用）
  add_cfg p5_iso_fedavg     isolated fedavg       ""
  add_cfg p5_iso_fedprox    isolated fedavg       fedprox
  add_cfg p5_iso_scaffold   isolated fedavg       scaffold
  add_cfg p5_iso_eda        isolated eda          ""
  add_cfg p5_iso_edaclu     isolated eda_cluster  ""
fi

# ==============================================================================
# 断点续跑（同一配置内续跑是安全的；**跨配置复用权重是绝对禁止的**）
# ==============================================================================
detect_start_round () {
  local root="$1" last="$2" d
  for ((d=last; d>=0; d--)); do
    if [ -f "$root/global_r${d}/trunk.pth" ]; then echo $((d + 1)); return 0; fi
  done
  echo 0
}

# 同构自检：日志首行必须是"从第 0 轮开始"
check_same_construction () {
  local root="$1" tag="$2"
  local lg="$LOGDIR/${tag}.log"
  [ -f "$lg" ] || return 0
  local first
  first=$(grep -m1 -oE "从第 [0-9]+ 轮开始" "$lg" 2>/dev/null | grep -oE "[0-9]+" || true)
  if [ -n "$first" ] && [ "$first" -gt 0 ]; then
    log "[!!] $tag 的训练是从第 $first 轮开始的 ⇒ **不是从 0 轮同构**（铁律 D）"
    log "    处理：清空 $root 后重跑该配置"
    return 1
  fi
  return 0
}

# ==============================================================================
# 评估
# ==============================================================================
eval_done () {
  local logf="$1"
  [ -f "$logf" ] || return 1
  local n
  n=$(grep -c '| avg' "$logf" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  [ "$n" -ge "$SEEDS" ]
}

eval_task () {
  local tag="$1" domain="$2" envname="$3" model_dir="$4"
  local logf="$OUT/eval_${tag}_${domain}.log"
  if [ ! -f "$model_dir/model.pth" ]; then
    log "[!!] $tag/$domain 缺 model.pth（$model_dir）—— 评估失败"
    return 1
  fi
  if eval_done "$logf"; then
    log "[SKIP] $tag/$domain 已评估"
    return 0
  fi
  log "[EVAL] $tag/$domain env=$envname seeds=$SEEDS episodes=$EPISODE_NUM"
  MODEL_DIR="$model_dir" DOMAIN="$domain" ENVNAME="$envname" \
    N_SEEDS="$SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    LOG_TAG="$tag" \
    bash "$SCRIPT_DIR/eval_model.sh" 2>&1 | tail -n 12

  # ===== 自验证：MoE 必须生效（参数量 > 18M），否则评的是残缺模型 =====
  local npm ok=0
  npm=$(grep -oE "number of params \(M\): [0-9.]+" "$logf" 2>/dev/null \
        | head -1 | grep -oE "[0-9.]+$" || true)
  if [ -z "$npm" ]; then
    log "[!!] $tag/$domain 日志里找不到参数量 —— 无法确认 MoE 是否生效"
  else
    awk -v v="$npm" 'BEGIN{ if (v > 18.0) exit 0; else exit 1 }' && ok=1
    if [ "$ok" = "1" ]; then
      log "[OK] $tag/$domain 参数量 ${npm}M（>18M，MoE 已生效）"
    else
      log "[!!] $tag/$domain 参数量仅 ${npm}M —— **MoE 没生效，评的是残缺模型**，本项作废"
      return 1
    fi
  fi
  if eval_done "$logf"; then
    log "[OK] $tag/$domain 评估完成"
    return 0
  fi
  log "[!!] $tag/$domain 评估未完成（表数=$(grep -c '| avg' "$logf" 2>/dev/null || echo 0)）"
  return 1
}

# ==============================================================================
# 单配置执行
# ==============================================================================
run_cfg () {
  local idx="$1"
  local tag="${CFG_TAGS[$idx]}" split="${CFG_SPLIT[$idx]}"
  local agg="${CFG_AGG[$idx]}" base="${CFG_BASE[$idx]}"
  local root="$OUT/$tag"
  local last=$((NUM_ROUNDS - 1))

  banner "$tag  划分=$split 聚合=$agg 基线=${base:-无}"

  # ---- 完成判据：训练产物 + 三任务评估齐全 ----
  if [ "$FORCE" != "1" ] && [ -f "$root/global_r${last}/trunk.pth" ]; then
    local all_ok=1
    for d in mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3; do
      eval_done "$OUT/eval_${tag}_${d}.log" || all_ok=0
    done
    if [ "$all_ok" = "1" ]; then
      log "[SKIP] $tag 已完成（训练+评估齐全）"
      return 0
    fi
    log "[WARN] $tag 训练产物在但评估不齐 —— 补评估"
  fi

  # ---- 同构自检：不是从 0 轮开始的 ⇒ 清空重跑 ----
  if ! check_same_construction "$root" "$tag"; then
    log "[FIX] 清空 $root 后从第 0 轮重跑 $tag"
    rm -rf "$root"
    rm -f "$LOGDIR/${tag}.log"
  fi

  # ---- 断点续跑（同一配置内；opt_state 必须在）----
  local sr; sr=$(detect_start_round "$root" "$last")
  if [ "$sr" -gt 0 ]; then
    local miss=0
    for ((c=0; c<4; c++)); do
      [ -f "$root/opt_state_c${c}.pt" ] || miss=1
    done
    if [ "$miss" = "1" ]; then
      log "[!!] $tag 缺 opt_state_c*.pt —— Adam 相位会错 ⇒ 清空重跑"
      rm -rf "$root"; rm -f "$LOGDIR/${tag}.log"; sr=0
    else
      log "[续跑] $tag 从第 $sr 轮继续（同一配置内续跑，lr 相位一致）"
    fi
  else
    log "[新跑] $tag 从第 0 轮开始"
  fi

  # ---- 划分环境变量 ----
  local env_extra=()
  if [ "$split" = "balanced" ]; then
    env_extra=(
      "CLIENT_DOMAINS_OVERRIDE=$BAL_DOMAINS"
      "CLIENT_ENVS_OVERRIDE=$BAL_ENVS"
      "CLIENT_EPOCH_SIZE_OVERRIDE=$BAL_ESIZE"
    )
  else
    env_extra=(
      "CLIENT_DOMAINS_OVERRIDE=$ISO_DOMAINS"
      "CLIENT_ENVS_OVERRIDE=$ISO_ENVS"
      "CLIENT_EPOCH_SIZE_OVERRIDE=$ISO_ESIZE"
    )
  fi

  # ---- 训练 ----
  ( cd /root/HPT && \
    env OUTPUT_ROOT="$root" \
        NUM_ROUNDS="$NUM_ROUNDS" \
        START_ROUND="$sr" \
        USE_GLOBAL_LR=1 \
        WARMUP_FRAC=0.25 \
        FEDVLA_PERSIST_OPT=1 \
        USE_MOE=1 \
        MOE_EXPERTS="$MOE_EXPERTS" \
        MOE_TOPK="$MOE_TOPK" \
        MOE_EXPERT_RATIO="$MOE_EXPERT_RATIO" \
        MOE_LAYERS="$MOE_LAYERS" \
        AGG_MODE="$agg" \
        BASELINE="$base" \
        "${env_extra[@]}" \
        bash "$SCRIPT_DIR/run_fedavg.sh" ) > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  if [ "$rc" != "0" ]; then
    log "[!!] $tag 训练失败 rc=$rc，日志尾部："; tail -n 25 "$LOGDIR/${tag}.log"
    return 1
  fi
  if [ ! -f "$root/global_r${last}/trunk.pth" ]; then
    log "[!!] $tag 训练结束但缺 global_r${last}/trunk.pth"; return 1
  fi

  # ---- 评估三个任务 ----
  local m="$root/global_r${last}"
  # SKIP_EVAL=1：冒烟测试用（只验训练侧机制，省掉每任务 ~35 分钟的评估）
  if [ "${SKIP_EVAL:-0}" = "1" ]; then
    log "[SMOKE] SKIP_EVAL=1 —— 跳过三个任务的评估"
    return 0
  fi
  eval_task "$tag" mujoco_metaworld_task1 reach-v2 "$m" || return 1
  eval_task "$tag" mujoco_metaworld_task2 push-v2 "$m" || return 1
  eval_task "$tag" mujoco_metaworld_task3 button-press-topdown-v2 "$m" || return 1

  # ---- 配置成功 ⇒ 回收中间产物（磁盘）----
  cleanup_completed_config "$root" "$tag" "$last"

  log "[OK] $tag 全部完成"
  return 0
}

# ==============================================================================
# 配置成功后的磁盘回收（2026-11-23 新增）
# ==============================================================================
# 🔴 为什么必须清：每个配置稳态占用实测 ~2.4GB
#      client_*_r*     4 客户端 × 176MB × 保留轮数      ≈ 1.4GB
#      opt_state_c*.pt 4 × 171MB（Adam 相位）            ≈ 0.68GB
#      global_r*       保留轮数 × 176MB                  ≈ 0.35GB
#    6 个新配置累计 ~14GB，而系统盘只剩 ~11GB ⇒ **必然爆盘**
#    （第一版 M2 只清了 `_baseline/`，没清这两个大头 —— 阻塞点 B 的第二层）
#
# 🔴 为什么删了不会破坏「补评估」续跑：
#    run_cfg 的完成判据（global_r{last}/trunk.pth 在 + 3 个评估日志齐全）
#    在 opt_state 检查**之前**就 return 0（SKIP）⇒
#    已经完整完成的配置，重跑时直接跳过，根本不会碰 opt_state；
#    只有「训练完但评估不齐」的配置才走续跑分支 —— 那种情况 opt_state 必须留着。
#    ⇒ 本函数**只在三个评估全部完成后**才调用。
#
# ⚠️ 保留 global_r{last}（评估与结果提取的对象），只删更早的轮次。
cleanup_completed_config () {
  local root="$1" tag="$2" last="$3"
  local n=0 f
  for f in "$root"/client_*_r*; do
    [ -e "$f" ] || continue
    rm -rf "$f" && n=$((n + 1))
  done
  for f in "$root"/opt_state_c*.pt; do
    [ -e "$f" ] || continue
    rm -f "$f" && n=$((n + 1))
  done
  # 只保留最后一轮的全局模型（更早的轮次不再需要）
  for f in "$root"/global_r*; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in
      "global_r${last}") continue ;;
    esac
    rm -rf "$f" && n=$((n + 1))
  done
  [ "$n" -gt 0 ] && log "[清理] $tag 已完成 ⇒ 回收 $n 个中间产物（client_* / opt_state / 早期 global_r）"
  df -h "$root" 2>/dev/null | tail -1 | sed 's/^/[清理后磁盘] /'
}

# ==============================================================================
# 主流程
# ==============================================================================
preflight

FAILED=()
for ((i=0; i<${#CFG_TAGS[@]}; i++)); do
  tag="${CFG_TAGS[$i]}"
  # ONLY_TAGS：只跑指定配置（冒烟测试用；默认空 = 全跑，行为不变）
  if [ -n "${ONLY_TAGS:-}" ] && ! echo " $ONLY_TAGS " | grep -q " $tag "; then
    log "[SKIP] $tag 不在 ONLY_TAGS（$ONLY_TAGS）里"
    continue
  fi
  if run_cfg "$i"; then
    echo 0 > "$OUT/.rc_${tag}"
  else
    echo 1 > "$OUT/.rc_${tag}"
    FAILED+=("$tag")
    log "[FAIL] $tag 失败 —— 继续跑下一个（最后统一报失败清单）"
  fi
done

banner "第 5 阶段汇总"
for t in "${CFG_TAGS[@]}"; do
  rc=$(cat "$OUT/.rc_${t}" 2>/dev/null || echo "?")
  if [ "$rc" = "0" ]; then echo "  ✅ $t"; else echo "  ❌ $t (rc=$rc)"; fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  log "[!!] 失败配置：${FAILED[*]}"
  log "    修复后可用同一命令续跑（已完成的会 SKIP）"
  exit 1
fi
log "全部 ${#CFG_TAGS[@]} 个配置完成 ✅"
