#!/usr/bin/env bash
# FedVLA 第 4 阶段：上行通信损伤扫描（丢包 + 量化压缩）。
#
# 目标（路线图 5.1）：三条"横轴通信参数、纵轴成功率"的曲线
#   · 丢包扫描：上行更新按固定比例丢失
#   · 压缩扫描：上行共享参数按不同比特量化
#   （流量预算曲线留到隐私侧一起做）
#
# 实验台选择：**异构划分 + MoE + 簇感知聚合**（= 第3阶段的最终方法 P3-C）
#   为什么不用 IID：
#     · IID 下数据同质，"少传一轮"影响本来就小，通信损伤测不出信号；
#     · 论文第4章要回答的是"**我们的方法**在链路受损下还站不站得住"，
#       必须用最终方法，否则通信曲线与 MoE/EDA 创新脱节，委员会会追问。
#   为什么看 task3：
#     · 异构下 task1/task2 的成功率被压到 0（地板效应），没有区分度；
#     · task3 是唯一"独占"的客户端所在域，基线成功率 0.45~0.95，有信号。
#     · 全部任务都评太慢（3 倍成本），可用 COMM_EVAL_TASKS=task1,task2,task3 扩展。
#
# 口径统一：
#   · 丢包用"固定丢 k 个 + 轮换"（fixed），**不用 Bernoulli**
#     —— 实测 Bernoulli 在 n=4 / rate=0.1 时会连续 50 轮一个都不丢，
#        损伤被随机性完全稀释，曲线会得出"丢包毫无影响"的错误结论。
#   · 量化复用 comm_analysis.py 的实现与误差口径（与门2 的通信成本表同源）。
#
# 幂等：每个配置独立目录 + detect_start_round 判据；已评估过的项按日志表数跳过。
# 磁盘：每个配置评估完立即删除该配置的权重目录（只留日志与结果），
#       否则 7 个配置 × 2.2G 会把 30G 的盘吃满（历史故障源）。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

NUM_ROUNDS_COMM=${NUM_ROUNDS_COMM:-25}     # 每个配置的轮数（比第3阶段的 50 短，为控预算）
SEEDS=${SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
N_JOBS=${N_JOBS:-3}
COMM_SEED=${COMM_SEED:-1}
KEEP_ROUNDS=${KEEP_ROUNDS:-2}
MIN_FREE_GB=${MIN_FREE_GB:-6}
# 评估哪些任务（默认只评 task3：唯一有信号的域）
COMM_EVAL_TASKS=${COMM_EVAL_TASKS:-task3}
# 评估完是否删除该配置的权重目录（默认删；设 0 保留）
CLEAN_AFTER_EVAL=${CLEAN_AFTER_EVAL:-1}

# ===== MoE 环境变量（训练与评估必须逐字一致）=====
# ⚠️ 铁律：patch_hpt_moe.py 只在 FEDVLA_MOE=1 时才构造 MoE 结构，
#    而 Policy.load_model 用 strict=False → 评估时若漏设，
#    264 个专家权重会被**静默丢弃**且零报错，评出来的是残缺模型。
#    实测：完整 MoE = 22.38M，残缺 = 13.94M。评估后必须核对参数量 > 18。
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half

# ===== 异构划分（与第3阶段 P3-C 完全一致）=====
HETERO_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1"
HETERO_ENVS="reach-v2 push-v2 button-press-topdown-v2 reach-v2"
HETERO_EPOCH_SIZE="15 19 24 15"

# ===== 配置表 =====
CONFIG_TAGS=( c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1 )
CONFIG_DROP=(  0       0.25     0.5      0.75     0       0       0      )
CONFIG_QUANT=( fp32    fp32     fp32     fp32     int8_g  int4_g  bit1   )

log () { echo "[$(date '+%F %T')] $*"; }

# ===== 修正评估脚本的线程数（自包含，幂等）=====
# 背景：eval_model.sh 里硬编码 `export OMP_NUM_THREADS=8`，而评估用
#   hydra multirun + joblib launcher（n_jobs=3）→ 3 个 worker × 8 线程 = 24 线程
#   压 15 核，实测 load average 冲到 22~29，评估慢一倍不止。
# 为什么在这里改而不是手改文件：第4阶段的守护 phase4_watch 会在第3阶段评估
#   一结束就立刻开跑，届时 eval_model.sh 可能正在被 bash 执行 ——
#   **改正在运行的脚本会让 bash 读到错乱字节**。
#   所以改成"可被环境变量覆盖"的形式，且只在它没被改动过时才动（幂等）。
EVAL_SH="$SCRIPT_DIR/eval_model.sh"
if [ -f "$EVAL_SH" ] && grep -q '^export OMP_NUM_THREADS=8' "$EVAL_SH"; then
  sed -i 's|^export OMP_NUM_THREADS=8$|export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"|' "$EVAL_SH"
  log "已把 eval_model.sh 的 OMP_NUM_THREADS 改为可覆盖（默认 2）"
fi
export OMP_NUM_THREADS=2

# 找出最后一个"完整跑完"的轮次 -> 下一轮起点
detect_start_round () {
  local root="$1" last="$2" d
  for ((d=last; d>=0; d--)); do
    if [ -f "$root/global_r${d}/trunk.pth" ]; then echo $((d + 1)); return 0; fi
  done
  echo 0
}

# 评估是否已完成（3 个 seed -> 3 张表）
eval_done () {
  local logf="$1" n
  [ -f "$logf" ] || return 1
  n=$(grep -c '| avg' "$logf" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  [ "$n" -ge "$SEEDS" ]
}

# 评估一个任务的模型目录
eval_task () {
  local tag="$1" domain="$2" envname="$3" model_dir="$4"
  local logf="$OUT/eval_${tag}_${domain}.log"
  if [ ! -f "$model_dir/model.pth" ]; then
    log "[SKIP] $tag/$domain 缺 model.pth（$model_dir）"
    return 1
  fi
  if eval_done "$logf"; then
    log "[SKIP] $tag/$domain 已评估（$(grep -c '| avg' "$logf" 2>/dev/null || true) 张表）"
    return 0
  fi
  log "[EVAL] $tag/$domain env=$envname seeds=$SEEDS episodes=$EPISODE_NUM moe=1"
  MODEL_DIR="$model_dir" DOMAIN="$domain" ENVNAME="$envname" \
    N_SEEDS="$SEEDS" EPISODE_NUM="$EPISODE_NUM" N_JOBS="$N_JOBS" \
    LOG_TAG="$tag" \
    bash "$SCRIPT_DIR/eval_model.sh" 2>&1 | tail -n 12

  # ===== 自验证：MoE 必须生效（参数量 > 18M），否则结果作废 =====
  if [ "$domain" = "mujoco_metaworld_task3" ]; then
    local npm
    npm=$(grep -oE "number of params \(M\): [0-9.]+" "$logf" 2>/dev/null \
          | head -1 | grep -oE "[0-9.]+$" || true)
    if [ -z "$npm" ]; then
      log "[!!] $tag/$domain 日志里找不到参数量 —— 无法确认 MoE 是否生效"
    else
      local ok=0
      awk -v v="$npm" 'BEGIN{ if (v > 18.0) exit 0; else exit 1 }' && ok=1
      if [ "$ok" = "1" ]; then
        log "[OK] $tag/$domain 参数量 ${npm}M（>18M，MoE 已生效）"
      else
        log "[!!] $tag/$domain 参数量仅 ${npm}M —— **MoE 没生效，评的是残缺模型**，本项作废"
      fi
    fi
  fi
  if eval_done "$logf"; then
    log "[OK] $tag/$domain 完成"
  else
    log "[!!] $tag/$domain 评估未完成（表数=$(grep -c '| avg' "$logf" 2>/dev/null || echo 0)）"
  fi
}

run_one () {
  local tag="$1" drop="$2" quant="$3"
  local root="$OUT/$tag"
  local last=$((NUM_ROUNDS_COMM - 1))

  # ---- 完成判据：**评估齐全** ----
  # ⚠️ 2026-11-08 修正：原来写的是
  #      [ -f "$root/global_r${last}/trunk.pth" ] && [ "$all_eval_done" = "1" ]
  #    但 CLEAN_AFTER_EVAL=1 会在评估完成后**删除整个权重目录** →
  #    实例一重启，已完成配置的 global_r24/trunk.pth 已不存在 → 判据失效 →
  #    该配置会被**从头重训 25 轮**（×7 配置 = 7 小时纯浪费），且日志被追加污染。
  #    正确判据：评估日志 eval_<tag>_<domain>.log 存在且 ≥SEEDS 张表。
  #    评估永远在训练之后 ⇒ "评估齐全" 蕴含 "训练完成"，是唯一可靠凭据。
  local all_eval_done=1 t d e model_dir
  for t in ${COMM_EVAL_TASKS//,/ }; do
    case "$t" in
      task1) d=mujoco_metaworld_task1; e=reach-v2 ;;
      task2) d=mujoco_metaworld_task2; e=push-v2 ;;
      task3) d=mujoco_metaworld_task3; e=button-press-topdown-v2 ;;
      *) continue ;;
    esac
    eval_done "$OUT/eval_${tag}_${d}.log" || { all_eval_done=0; break; }
  done
  if [ "$all_eval_done" = "1" ]; then
    log "[SKIP] $tag 已完成（评估齐全：$OUT/eval_${tag}_*.log）"
    return 0
  fi

  # ---- 训练 ----
  local start
  start=$(detect_start_round "$root" "$last")
  log "===== $tag 训练（drop=$drop quant=$quant 轮数=$NUM_ROUNDS_COMM 从第 $start 轮开始）====="
  OUTPUT_ROOT="$root" NUM_ROUNDS="$NUM_ROUNDS_COMM" \
  CLIENT_DOMAINS_OVERRIDE="$HETERO_DOMAINS" \
  CLIENT_ENVS_OVERRIDE="$HETERO_ENVS" \
  CLIENT_EPOCH_SIZE_OVERRIDE="$HETERO_EPOCH_SIZE" \
  USE_MOE=1 AGG_MODE=eda_cluster \
  COMM_DROP_RATE="$drop" COMM_QUANT_SCHEME="$quant" COMM_SEED="$COMM_SEED" \
  KEEP_ROUNDS="$KEEP_ROUNDS" MIN_FREE_GB="$MIN_FREE_GB" \
  START_ROUND="$start" \
    bash "$SCRIPT_DIR/run_fedavg.sh" >> "$OUT/log_${tag}_train.txt" 2>&1

  if [ ! -f "$root/global_r${last}/trunk.pth" ]; then
    log "[!!] $tag 训练未完成（缺 global_r${last}/trunk.pth），查看 $OUT/log_${tag}_train.txt"
    return 1
  fi
  log "$tag 训练完成"

  # ---- 评估 ----
  for t in ${COMM_EVAL_TASKS//,/ }; do
    case "$t" in
      task1) d=mujoco_metaworld_task1; e=reach-v2 ;;
      task2) d=mujoco_metaworld_task2; e=push-v2 ;;
      task3) d=mujoco_metaworld_task3; e=button-press-topdown-v2 ;;
      *) continue ;;
    esac
    model_dir="$root/global_r${last}"
    eval_task "$tag" "$d" "$e" "$model_dir"
  done

  # ---- 磁盘回收：删除该配置的权重目录，只留日志 ----
  if [ "$CLEAN_AFTER_EVAL" = "1" ]; then
    for g in "$root"/global_r* "$root"/client_*_r*; do
      [ -d "$g" ] && rm -rf "$g"
    done
    # opt_state 是 Adam 状态，训练结束即无用
    rm -f "$root"/opt_state_c*.pt
    log "$tag 权重已清理（保留日志与结果）"
  fi
  df -h "$OUT" | tail -1 | sed 's/^/[磁盘] /'
}

echo "=============================================================="
echo "FedVLA 第4阶段：通信损伤扫描"
echo "轮数=$NUM_ROUNDS_COMM  配置数=${#CONFIG_TAGS[@]}  评估任务=$COMM_EVAL_TASKS"
echo "丢包口径=fixed（每轮固定丢 k 个 + 轮换）  量化口径=comm_analysis.py 同源"
echo "=============================================================="

for i in "${!CONFIG_TAGS[@]}"; do
  run_one "${CONFIG_TAGS[$i]}" "${CONFIG_DROP[$i]}" "${CONFIG_QUANT[$i]}"
done

echo "=============================================================="
echo "第4阶段通信损伤扫描全部处理完毕"
echo "=============================================================="
