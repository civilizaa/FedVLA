#!/usr/bin/env bash
# FedVLA 第4阶段：**DP 噪声 vs 真实任务成功率**扫描。
#
# 为什么必须做这个实验：
#   第4阶段隐私侧目前只有**代理指标**（共享主干保真度 / 重建相似度），
#   论文若只报代理指标，委员会一定会问"那真实任务还能不能学会？"
#   ⇒ 必须补上**真实任务成功率**随 DP 噪声的曲线，做交叉验证。
#
# 实验台：与通信扫描**完全一致**的 P3-C 配置（异构划分 + MoE + 簇感知聚合，50 轮），
#   唯一变量是上行 DP 噪声 σ ⇒ 干净基线可直接复用通信扫描的 c_drop00。
#
# σ 的语义（本脚本的核心）：
#   上行加噪形式  Δ·min(1, C/‖Δ‖₂) + N(0, (σ·C)²)（**per-coordinate** 噪声），
#   C = ‖Δ‖₂ 中位数（标定得到）。
#   ⇒ 总噪声 L2 = σ·C·√d，信号 L2 = C ⇒ **噪声/信号比 = σ·√d**。
#   FedVLA 上行 d ≈ 17.1M ⇒ √d ≈ 4134
#     ⇒ σ=1e-4 → 噪声/信号 0.41（与信号同量级，**转折点**）
#     ⇒ σ=1e-3 → 4.1 倍；σ=1e-2 → 41 倍（噪声完全主导）
#   ⚠️ 这与隐私侧 dp.py 的 "噪声总范数 = σ·C·√(17.1M) = σ·4139" 是同一件事，
#      所以本扫描的 σ 网格**必须与隐私侧完全一致**，两边才能在同一个 σ 上对表：
#        真实成功率  vs  代理指标（共享主干保真度 / 动作重建 cos）。
#   ε 只依赖 σ（δ=1e-5, q=1/4, 50 轮复合），与 C 无关 ⇒ 两边的 ε 也直接可比。
#
# σ 网格：{1e-4, 1e-3, 1e-2}
#   · 1e-4：噪声与信号同量级 ⇒ 期望"明显受损但还能学"
#   · 1e-3：噪声 4 倍于信号 ⇒ 期望大幅退化
#   · 1e-2：噪声 41 倍 ⇒ 期望落到地板（成功率 ≈ 0）
#   这三个点与隐私侧的 uniform 方案 σ 网格完全重合 ⇒ 交叉验证成立。
#
# 幂等：每个配置独立目录 + 评估日志表数判据；已评估过的配置跳过。
# 磁盘：评估完立即删该配置权重目录（30G 盘，3 配置 × 2.2G 会吃满）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

NUM_ROUNDS=${NUM_ROUNDS:-50}          # 与 P3-C / 通信扫描对齐
SEEDS=${SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-2}
MAX_PAR=${MAX_PAR:-3}
KEEP_ROUNDS=${KEEP_ROUNDS:-1}
MIN_FREE_GB=${MIN_FREE_GB:-4}
DP_SEED=${DP_SEED:-1234}
# 可选：加跑一个 σ=0 的配置做"管线一致性自检"（应与 c_drop00 的 0.667 接近）
DP_CHECK_BASELINE=${DP_CHECK_BASELINE:-0}

# ===== MoE 环境变量（训练与评估必须逐字一致）=====
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half

# ===== 异构划分（与 P3-C 完全一致）=====
HETERO_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1"
HETERO_ENVS="reach-v2 push-v2 button-press-topdown-v2 reach-v2"
HETERO_EPOCH_SIZE="15 19 24 15"

log () { echo "[$(date '+%F %T')] $*"; }

# ===== C 标定结果 =====
CALIB="$OUT/dp_calib/calib.json"
if [ ! -f "$CALIB" ]; then
  echo "[!!] 找不到标定结果 $CALIB —— 先跑 calibrate_dp_clip.sh"
  exit 1
fi
DP_CLIP_NORM=$("$HPT_PY" -c "import json;print(json.load(open('$CALIB'))['clip_norm_C'])")
log "C（‖Δ‖₂ 中位数）= $DP_CLIP_NORM"

# ===== σ 配置表 =====
#   tag : 目录名（eval_<tag>_*.log 是完成判据，不能含点号）
#   sigma : 噪声倍率（DP 语义：per-coordinate std = σ·C）
DP_TAGS=( dp_s0001 dp_s0003 dp_s0012 )
DP_SIGMAS=( 1e-4    1e-3    1e-2   )
if [ "$DP_CHECK_BASELINE" = "1" ]; then
  DP_TAGS=( dp_s0000 dp_s0001 dp_s0003 dp_s0012 )
  DP_SIGMAS=( 0.0     1e-4    1e-3    1e-2  )
fi

eval_done () {
  local logf="$1" n
  [ -f "$logf" ] || return 1
  n=$(grep -c '| avg' "$logf" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  [ "$n" -ge "$SEEDS" ]
}

run_one () {
  local tag="$1" sigma="$2"
  local root="$OUT/$tag"
  local last=$((NUM_ROUNDS - 1))
  local logf="$OUT/eval_${tag}_mujoco_metaworld_task3.log"

  # ---- 完成判据：评估齐全（权重目录评估后会被删，不能用权重当判据）----
  if eval_done "$logf"; then
    log "[SKIP] $tag 已完成（评估齐全）"
    return 0
  fi

  # ---- 训练（带上行 DP 加噪）----
  log "===== $tag 训练（σ=$sigma, C=$DP_CLIP_NORM, 轮数=$NUM_ROUNDS）====="
  OUTPUT_ROOT="$root" NUM_ROUNDS="$NUM_ROUNDS" \
  CLIENT_DOMAINS_OVERRIDE="$HETERO_DOMAINS" \
  CLIENT_ENVS_OVERRIDE="$HETERO_ENVS" \
  CLIENT_EPOCH_SIZE_OVERRIDE="$HETERO_EPOCH_SIZE" \
  USE_MOE=1 AGG_MODE=eda_cluster \
  POST_CLIENT_HOOK=dp_noise_client.py \
  DP_SIGMA="$sigma" DP_CLIP_NORM="$DP_CLIP_NORM" DP_SEED="$DP_SEED" \
  KEEP_ROUNDS="$KEEP_ROUNDS" MIN_FREE_GB="$MIN_FREE_GB" \
    bash "$SCRIPT_DIR/run_fedavg.sh" >> "$OUT/log_${tag}_train.txt" 2>&1

  if [ ! -f "$root/global_r${last}/trunk.pth" ]; then
    log "[!!] $tag 训练未完成（缺 global_r${last}/trunk.pth），查看 $OUT/log_${tag}_train.txt"
    return 1
  fi
  log "$tag 训练完成"

  # ---- 评估 task3（唯一有信号的域）----
  if [ ! -f "$root/global_r${last}/model.pth" ]; then
    log "[!!] $tag 缺 model.pth（$root/global_r${last}）"
    return 1
  fi
  log "[EVAL] $tag task3 seeds=$SEEDS episodes=$EPISODE_NUM moe=1"
  MODEL_DIR="$root/global_r${last}" DOMAIN="mujoco_metaworld_task3" \
  ENVNAME="button-press-topdown-v2" \
  N_SEEDS="$SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
  LOG_TAG="$tag" \
    bash "$SCRIPT_DIR/eval_model.sh" 2>&1 | tail -n 12

  # ===== 自验证：MoE 必须生效（参数量 > 18M），否则结果作废 =====
  local npm ok=0
  npm=$(grep -oE "number of params \(M\): [0-9.]+" "$logf" 2>/dev/null \
        | head -1 | grep -oE "[0-9.]+$" || true)
  if [ -z "$npm" ]; then
    log "[!!] $tag 日志里找不到参数量 —— 无法确认 MoE 是否生效"
  else
    awk -v v="$npm" 'BEGIN{ if (v > 18.0) exit 0; else exit 1 }' && ok=1
    if [ "$ok" = "1" ]; then
      log "[OK] $tag 参数量 ${npm}M（>18M，MoE 已生效）"
    else
      log "[!!] $tag 参数量仅 ${npm}M —— **MoE 没生效，评的是残缺模型**，本项作废"
    fi
  fi

  if eval_done "$logf"; then
    log "[OK] $tag 评估完成"
  else
    log "[!!] $tag 评估未完成（表数=$(grep -c '| avg' "$logf" 2>/dev/null || echo 0)）"
  fi

  # ---- 磁盘回收 ----
  for g in "$root"/global_r* "$root"/client_*_r*; do
    [ -d "$g" ] && rm -rf "$g"
  done
  rm -f "$root"/opt_state_c*.pt
  log "$tag 权重已清理（保留日志与结果）"
  df -h "$OUT" | tail -1 | sed 's/^/[磁盘] /'
}

echo "=============================================================="
echo "FedVLA 第4阶段：DP 噪声 vs 真实任务成功率"
echo "σ 网格 = ${DP_SIGMAS[*]}   C = $DP_CLIP_NORM   轮数 = $NUM_ROUNDS"
echo "实验台 = P3-C（异构 + MoE + 簇感知），评估 task3 × ${SEEDS} seeds × ${EPISODE_NUM} rollouts"
echo "=============================================================="

FAILED_TAGS=()
running=0
for i in "${!DP_TAGS[@]}"; do
  tag="${DP_TAGS[$i]}"
  rm -f "$OUT/.rc_${tag}"
  (
    run_one "$tag" "${DP_SIGMAS[$i]}" > "$OUT/dplog_${tag}.log" 2>&1
    echo $? > "$OUT/.rc_${tag}"
  ) &
  running=$((running + 1))
  log "已启动配置 ${tag}（σ=${DP_SIGMAS[$i]}，第 ${running}/${MAX_PAR} 路并行中）"
  if [ "$running" -ge "$MAX_PAR" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

for tag in "${DP_TAGS[@]}"; do
  rc=1
  [ -f "$OUT/.rc_${tag}" ] && rc=$(cat "$OUT/.rc_${tag}")
  if [ "$rc" != "0" ]; then
    FAILED_TAGS+=("$tag")
    log "[!!] 配置 ${tag} 未完成（rc=${rc}，见 $OUT/dplog_${tag}.log）"
  else
    log "[OK] 配置 ${tag} 完成"
  fi
done

if [ "${#FAILED_TAGS[@]}" -gt 0 ]; then
  echo "=============================================================="
  log "[!!] DP 效用扫描**未全部完成**：${FAILED_TAGS[*]}"
  log "     修好问题后重跑本脚本即可续跑（幂等）"
  echo "=============================================================="
  exit 1
fi

echo "=============================================================="
log "DP 效用扫描全部处理完毕"
echo "=============================================================="
