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
#
# ============================ v2（2026-11-08）============================
# v2 相对 v1 的三处改动：
#   1) 轮数 25 → 50（**关键修正**）：v1 的 25 轮下基线 c_drop00 在 task3 上只有
#      0.017，c_int8 是 0.000 —— 基线落在地板上，曲线零区分度。
#      50 轮与第3阶段 P3-C 对齐后，P3-C 在 task3 上实测 0.60/0.50，有信号。
#   2) 串行 → **3 路并行**（RTX 3090 24G，单配置训练占 5.4G），
#      7 配置 × 120 分钟 串行要 14 小时，并行后约 5 小时。
#   3) 启动时把 **25 轮的旧评估/训练日志改名备份**（*.r25），
#      否则"评估齐全"这个幂等判据会把 25 轮的旧结果当成 50 轮的结果 ——
#      又是一次 silent failure（旧结果 < 新基线，曲线会整体偏低）。
# ==========================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

# ⚠️ 2026-11-08 v2：轮数 25 → 50
#   原因（实测硬伤）：25 轮下 **基线本身就没学会**。
#     · P3-C（= 本扫描的 c_drop00 配置）50 轮 → task3 成功率 0.60/0.50
#     · 本扫描 25 轮 → c_drop00 task3 = 0.05/0.00/0.00（均值 0.017），c_int8 = 0.000
#   ⇒ 所有配置的**基线落在地板上**，丢包/量化曲线没有任何区分度，
#     这样的曲线写进论文是错的（会得出"任何损伤都无差别"的假结论）。
#   ⇒ 轮数必须与第3阶段 P3-C 对齐到 50，才有可比的绝对成功率。
NUM_ROUNDS_COMM=${NUM_ROUNDS_COMM:-50}     # 与第3阶段 P3-C 对齐（50 轮）
SEEDS=${SEEDS:-3}
EPISODE_NUM=${EPISODE_NUM:-20}
# ⚠️ 3 路并行时，评估 worker 总数 = 并行配置数 × N_JOBS。
#   机器 15 核，每个 worker 2 线程 → N_JOBS=2 时峰值 12 线程，可接受。
N_JOBS=${N_JOBS:-2}
# ===== v2：评估串行锁 =====
# 为什么需要：3 路训练并行时，若某配置先跑完就立刻进入评估，
#   峰值显存 = 2 路训练 (10.8G) + 3 配置 × 2 workers 的 MuJoCo (≈12G)
#   → 逼近 3090 的 24G，可能 OOM 导致评估崩在中间（又是 silent failure）。
#   所以评估**全局串行**，一次只跑一个配置的评估，训练照常并行。
#   死锁检测：实例重启会留下锁目录（持锁进程已死），
#   抢锁前读 pid 用 kill -0 判活，死锁则清掉重抢（2026-11-07 的教训）。
COMM_SEED=${COMM_SEED:-1}
# 并行下每配置只留 1 轮权重 → 单配置峰值约 1.2G，3 路约 3.6G（30G 盘）
KEEP_ROUNDS=${KEEP_ROUNDS:-1}
MIN_FREE_GB=${MIN_FREE_GB:-4}
# 并行度：RTX 3090 24GB，单配置训练占约 5.4GB → 3 路安全
MAX_PAR=${MAX_PAR:-3}
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
# ⚠️ 2026-11-19：支持用环境变量 COMM_TAGS 指定**要跑的配置子集**。
#   默认 = 全部 7 个（与历史行为一致）。
#   修正版重跑时由外层脚本显式传 COMM_TAGS，保证只跑需要的配置，
#   而且**保留配置（c_drop50/c_drop75）也要在列表里** ——
#   它们会被"评估齐全"判据 SKIP 掉，同时让 .rc_<tag> 正常写出，
#   下游 chain_phase4_dp.sh 的"7 个 .rc 齐全"判据才不会卡死。
ALL_TAGS=( c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1 )
CONFIG_TAGS=()
CONFIG_DROP=()
CONFIG_QUANT=()
if [ -n "${COMM_TAGS:-}" ]; then
  SELECTED="$COMM_TAGS"
else
  SELECTED="${ALL_TAGS[*]}"
fi
for _t in $SELECTED; do
  case "$_t" in
    c_drop00) _d=0;    _q=fp32  ;;
    c_drop25) _d=0.25; _q=fp32  ;;
    c_drop50) _d=0.5;  _q=fp32  ;;
    c_drop75) _d=0.75; _q=fp32  ;;
    c_int8)   _d=0;    _q=int8_g ;;
    c_int4)   _d=0;    _q=int4_g ;;
    c_bit1)   _d=0;    _q=bit1   ;;
    *) echo "[FATAL] 未知的配置标签：$_t"; exit 2 ;;
  esac
  CONFIG_TAGS+=("$_t"); CONFIG_DROP+=("$_d"); CONFIG_QUANT+=("$_q")
done

log () { echo "[$(date '+%F %T')] $*"; }

# ===== v2：备份 25 轮的旧日志（一次性）=====
# ⚠️ 为什么要备份而不是直接删：
#   · 25 轮的结果虽然基线在地板上，但它是"通信损伤 vs 训练轮数"的对照数据，
#     也是这次 silent failure 的实证记录，论文的"实验设计修正"一节要引用。
#   · 幂等判据 eval_done() 数的是日志里 '| avg' 的行数，
#     不备份的话 50 轮还没跑，脚本就会认为该配置"已评估完成"而跳过 ⇒ 结果错配。
if [ ! -f "$OUT/.r25_logs_backed_up" ]; then
  for tag in "${CONFIG_TAGS[@]}"; do
    for d in mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3; do
      f="$OUT/eval_${tag}_${d}.log"
      [ -f "$f" ] && mv "$f" "${f}.r25"
    done
    t="$OUT/log_${tag}_train.txt"
    [ -f "$t" ] && mv "$t" "${t}.r25"
  done
  touch "$OUT/.r25_logs_backed_up"
  log "已把 25 轮的旧评估/训练日志改名备份为 *.r25（防幂等判据误判为已完成）"
fi

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

# ===== v2：评估串行锁（带死锁检测）=====
EVAL_LOCK="$OUT/.eval_lock"
eval_lock_acquire () {
  local waited=0 pid=""
  while : ; do
    if mkdir "$EVAL_LOCK" 2>/dev/null; then
      echo $$ > "$EVAL_LOCK/pid"
      return 0
    fi
    pid=""
    [ -f "$EVAL_LOCK/pid" ] && pid=$(cat "$EVAL_LOCK/pid" 2>/dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      log "[锁] 发现死锁（持锁 pid=$pid 已不存在），清理后重抢"
      rm -rf "$EVAL_LOCK"
      continue
    fi
    sleep 20
    waited=$((waited + 1))
    if [ "$waited" -gt 270 ]; then
      log "[锁] 等待超过 90 分钟仍未拿到，强制清理后重抢"
      rm -rf "$EVAL_LOCK"
    fi
  done
}
eval_lock_release () { rm -rf "$EVAL_LOCK"; }

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
  # ⚠️ 2026-11-08 18:30 修正：**去掉全局评估串行锁**。
  #   原设计担心"3 路训练 + 多配置评估叠加导致显存 OOM"，实测：
  #     · 训练进程显存 4.3G/路，3 路共 13G，剩余 10.6G；
  #     · MuJoCo 评估是 **CPU 渲染 + 小模型 GPU 推理**（22M 模型 < 1G/worker），
  #       3 配置 × 2 workers ≈ 6G，合计 ~19G < 24G，安全；
  #     · 各配置的评估输出目录 `eval_<tag>_<domain>` **天然隔离**，不会互相写坏。
  #   而串行评估要额外花 7 配置 × 30 分钟 ≈ 3.5 小时（并行只需 ~1.2 小时）。
  #   eval_lock_acquire/release 函数保留（万一以后要退回串行模式）。
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
echo "FedVLA 第4阶段：通信损伤扫描 (v2, ${NUM_ROUNDS_COMM} 轮, ${MAX_PAR} 路并行)"
echo "轮数=$NUM_ROUNDS_COMM  配置数=${#CONFIG_TAGS[@]}  评估任务=$COMM_EVAL_TASKS"
echo "丢包口径=fixed（每轮固定丢 k 个 + 轮换）  量化口径=comm_analysis.py 同源"
echo "=============================================================="

# ===== v2：并行调度 =====
# 每个配置一个后台子进程，日志各自独立（scanlog_<tag>.log）。
# 退出码写到 $OUT/.rc_<tag>，主进程最后统一判定失败项 —— 因为
#   后台子进程的返回码不能直接从 wait 拿到（wait -n 只给状态，不给是哪个），
#   用文件传递退出码才不会把"哪个配置失败"弄丢。
FAILED_TAGS=()
running=0
for i in "${!CONFIG_TAGS[@]}"; do
  tag="${CONFIG_TAGS[$i]}"
  rm -f "$OUT/.rc_${tag}"
  (
    run_one "$tag" "${CONFIG_DROP[$i]}" "${CONFIG_QUANT[$i]}" \
      > "$OUT/scanlog_${tag}.log" 2>&1
    echo $? > "$OUT/.rc_${tag}"
  ) &
  running=$((running + 1))
  log "已启动配置 ${tag}（第 ${running}/${MAX_PAR} 路并行中）"
  if [ "$running" -ge "$MAX_PAR" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait

for tag in "${CONFIG_TAGS[@]}"; do
  rc=1
  [ -f "$OUT/.rc_${tag}" ] && rc=$(cat "$OUT/.rc_${tag}")
  if [ "$rc" != "0" ]; then
    FAILED_TAGS+=("$tag")
    log "[!!] 配置 ${tag} 未完成（rc=${rc}，见 $OUT/scanlog_${tag}.log）"
  else
    log "[OK] 配置 ${tag} 完成"
  fi
done

if [ "${#FAILED_TAGS[@]}" -gt 0 ]; then
  echo "=============================================================="
  log "[!!] 第4阶段通信扫描**未全部完成**：${FAILED_TAGS[*]}"
  log "     修好问题后重跑本脚本即可续跑（幂等：已完成的配置按评估日志跳过）"
  echo "=============================================================="
  exit 1
fi

echo "=============================================================="
echo "第4阶段通信损伤扫描全部处理完毕"
echo "=============================================================="
