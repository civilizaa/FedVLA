#!/usr/bin/env bash
# FedVLA 联邦训练编排（FedAvg）：4 客户端串行，50 轮，每轮本地训后聚合 trunk。
#
# 设计（已在 README 论证）：
#   - 复用 HPT 自带训练 `python -m hpt.run`，不重写训练循环。
#   - 第 0 轮各客户端从 INIT_PRETRAINED(hf) 训 3 epoch -> 各自 model.pth。
#   - 聚合：aggregate.py 平均 trunk -> 全局目录的 trunk.pth + model.pth + config*。
#   - 第 r+1 轮各客户端从全局目录续训（HPT init_policy 靠 trunk.pth 走本地续训分支）。
#   - 数据离线 Zarr 缓存，首轮落地后复用，不每轮重采。
#
# 用法：
#   bash run_fedavg.sh                # 从第 0 轮跑满 NUM_ROUNDS
#   START_ROUND=10 bash run_fedavg.sh  # 从指定轮续跑（断点续训）
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/fedavg_config.sh"

# ===== 环境（绝对路径 + 显式 export，避免新 shell 里 conda 失效）=====
export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

START_ROUND=${START_ROUND:-0}

# ===== 磁盘水位守卫 =====
# 背景（2026-11-03 实测踩坑）：每轮每客户端都产出 model.pth(66MB) + trunk.pth(50MB)，
# 50 轮 × 4 客户端 ≈ 250 个目录 ≈ 23GB —— 把 30GB 的系统盘直接吃满，
# 结果是 agent 心跳停摆、正在跑的评估被写日志失败打断（已实测）。
# 所以：默认只保留最近 KEEP_ROUNDS 轮的权重产物，日志一律保留
#（收敛曲线 extract_convergence.py 靠 log_client_*.txt，不受影响）。
KEEP_ROUNDS=${KEEP_ROUNDS:-2}      # 保留最近 N 轮的 client/global 权重
MIN_FREE_GB=${MIN_FREE_GB:-6}       # 可用空间低于此值时强制只留最近 1 轮

free_gb () {
  local v
  v=$(df -BG --output=avail "$OUTPUT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
  echo "${v:-999}"   # df 不可用时返回 999，不触发强制清理
}

# 删除第 $1 轮之前（不含）的 client_*/global_* 权重目录；第 $2 个参数是保留轮数
cleanup_old_rounds () {
  local cur="$1" keep="$2"
  local cutoff=$((cur - keep))
  [ "$cutoff" -lt 1 ] && return 0
  local targets=()
  for ((d=0; d<cutoff; d++)); do
    targets+=("$OUTPUT_ROOT/global_r$d")
    for ((c=0; c<NUM_CLIENTS; c++)); do
      targets+=("$OUTPUT_ROOT/client_${c}_r$d")
    done
  done
  local existed=0 sz=0
  for t in "${targets[@]}"; do
    if [ -d "$t" ]; then
      existed=$((existed+1))
      sz=$((sz + $(du -sm "$t" 2>/dev/null | cut -f1)))
    fi
  done
  if [ "$existed" -gt 0 ]; then
    rm -rf "${targets[@]}"
    echo "[清理] 删除 $existed 个历史轮次权重目录，释放约 ${sz} MB（保留最近 ${keep} 轮 + 全部日志）"
  fi
  df -h "$OUTPUT_ROOT" | tail -1 | sed 's/^/[清理后磁盘] /'
}

echo "===== FedVLA FedAvg ====="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "NUM_ROUNDS=$NUM_ROUNDS  NUM_CLIENTS=$NUM_CLIENTS  LOCAL_EPOCHS=$LOCAL_EPOCHS"
echo "CLIENT_DOMAINS: ${CLIENT_DOMAINS[*]}"
echo "CLIENT_ENVS: ${CLIENT_ENVS[*]}"
echo "CLIENT_EPOCH_SIZE: ${CLIENT_EPOCH_SIZE[*]}"
echo "从轮 $START_ROUND 开始"
echo "========================="

mkdir -p "$OUTPUT_ROOT"

# ===== 开训前硬拦截：load_trunk 补丁必须存在 =====
# 不打补丁的后果是"静默"的：客户端每轮都从 hf 原始权重重新开始，
# 联邦退化成"每轮独立训练"，聚合的权重根本没被用上 —— 不报错，结果全废。
# 所以在烧 50 轮算力之前必须拦下来。
POLICY_FILE="${HPT_DIR:-/root/HPT}/hpt/models/policy.py"
if [ ! -f "$POLICY_FILE" ]; then
  echo "[!!] 找不到 $POLICY_FILE"
  exit 1
fi
if ! grep -q "FedVLA patch" "$POLICY_FILE"; then
  echo "[!!] 检测到 HPT 的 load_trunk 未打补丁 —— 联邦续训会静默失效！"
  echo "     请先执行：bash $SCRIPT_DIR/patch_inline.sh"
  echo "     （或 python $SCRIPT_DIR/patch_hpt_load_trunk.py）"
  exit 1
fi
echo "[OK] load_trunk 补丁已就位"

# ===== 全局步数感知调度（可选，默认开启）=====
# 背景：HPT 原生把 cosine 的 T_max 硬编码为 train.total_iters，联邦下这是
# 【单轮本地步数】；再加上每轮一个新进程（scheduler/optimizer 重建），
# 于是 50 轮 = 50 次 "warmup -> cosine -> 0" 的 LR 重启，且 Adam 动量每轮清零。
# 补丁让 lr 成为【全局进度 p】的函数，联邦 50 轮的 lr 轨迹与同预算集中式逐点相同。
USE_GLOBAL_LR=${USE_GLOBAL_LR:-1}
WARMUP_FRAC=${WARMUP_FRAC:-0.25}
if [ "$USE_GLOBAL_LR" = "1" ]; then
  SCHED_FILE="${HPT_DIR:-/root/HPT}/hpt/utils/fedvla_lr.py"
  if [ ! -f "$SCHED_FILE" ]; then
    echo "[!!] 找不到 $SCHED_FILE —— 全局调度补丁未打！"
    echo "     请先执行：python $SCRIPT_DIR/patch_hpt_global_lr.py"
    echo "     （或用 USE_GLOBAL_LR=0 退回原生调度，但会重现 50 次 LR 重启）"
    exit 1
  fi
  if ! grep -q "GlobalStepLRScheduler" "${HPT_DIR:-/root/HPT}/hpt/run.py"; then
    echo "[!!] hpt/run.py 里没有 GlobalStepLRScheduler —— 补丁未注入！"
    exit 1
  fi
  echo "[OK] 全局步数感知调度补丁已就位（warmup_frac=$WARMUP_FRAC）"
else
  echo "[WARN] USE_GLOBAL_LR=0 —— 使用 HPT 原生调度（每轮重启 warmup+cosine）"
  unset FEDVLA_GLOBAL_LR FEDVLA_P_START FEDVLA_P_END FEDVLA_NUM_STEPS \
        FEDVLA_WARMUP_FRAC FEDVLA_OPT_STATE
fi

# ===== MoE（top-k 专家混合）开关 =====
# 第 3 阶段的实验开关。默认关闭（保持与门2 完全一致的基线行为）。
# 开启后：
#   - 客户端在 HPT trunk 的后半层 block 上挂「稠密 FFN + 并联零初始化 MoE 残差」
#     （训练起点严格等于预训练模型，预训练权重原样载入）
#   - 每轮本地训练结束导出 expert_counts.pt（专家选择计数）
#   - 聚合可选 EDA（逐层按计数相似度加权）—— 见 AGG_MODE
USE_MOE=${USE_MOE:-0}
MOE_EXPERTS=${MOE_EXPERTS:-8}
MOE_TOPK=${MOE_TOPK:-2}
MOE_EXPERT_RATIO=${MOE_EXPERT_RATIO:-0.25}
MOE_LAYERS=${MOE_LAYERS:-half}
if [ "$USE_MOE" = "1" ]; then
  HPT_ROOT="${HPT_DIR:-/root/HPT}"
  if [ ! -f "$HPT_ROOT/hpt/utils/fedvla_moe.py" ]; then
    echo "[!!] 找不到 $HPT_ROOT/hpt/utils/fedvla_moe.py —— MoE 补丁未打！"
    echo "     请先执行：python $SCRIPT_DIR/patch_hpt_moe.py"
    exit 1
  fi
  if ! grep -q "FedVLA-MOE" "$HPT_ROOT/hpt/models/transformer.py"; then
    echo "[!!] transformer.py 里没有 FedVLA-MOE 标记 —— MoE 补丁未注入！"
    exit 1
  fi
  if ! grep -q "FedVLA-MOE" "$HPT_ROOT/hpt/models/policy.py"; then
    echo "[!!] policy.py 里没有 FedVLA-MOE 标记 —— load_trunk 的 MoE 适配未注入！"
    echo "     注意：MoE 补丁必须打在 load_trunk 补丁之后"
    echo "     请先执行：python $SCRIPT_DIR/patch_hpt_load_trunk.py"
    echo "               python $SCRIPT_DIR/patch_hpt_moe.py"
    exit 1
  fi
  export FEDVLA_MOE=1
  export FEDVLA_MOE_EXPERTS="$MOE_EXPERTS"
  export FEDVLA_MOE_TOPK="$MOE_TOPK"
  export FEDVLA_MOE_EXPERT_RATIO="$MOE_EXPERT_RATIO"
  export FEDVLA_MOE_LAYERS="$MOE_LAYERS"
  echo "[OK] MoE 已开启（E=$MOE_EXPERTS k=$MOE_TOPK ratio=$MOE_EXPERT_RATIO layers=$MOE_LAYERS）"
fi

# ===== 聚合模式：fedavg | eda | eda_cluster =====
# fedavg    ：所有层用同一组权重（等权或按样本数）
# eda       ：逐层用 EDA 中心性权重（来自专家选择计数的余弦相似度）
# eda_cluster：簇感知 —— 先分簇，簇内中心性加权 + 簇间等权（H2 的执行路径）
AGG_MODE=${AGG_MODE:-fedavg}
if [ "$AGG_MODE" != "fedavg" ]; then
  if [ "$USE_MOE" != "1" ]; then
    echo "[!!] AGG_MODE=$AGG_MODE 需要 USE_MOE=1（EDA 的信号来自专家计数）"
    exit 1
  fi
  echo "[OK] 聚合模式 = $AGG_MODE"
fi

# ===== 开训前硬拦截：全局目录必须有 extras.pth =====
# extras.pth 存 modalities_tokens（policy 顶层 ParameterDict，不在 trunk 内）。
# 缺失时续训的模态 token 保持随机初始化，共享 trunk 会收到不一致输入。
for ((rr=1; rr<NUM_ROUNDS; rr++)); do
  GD="$OUTPUT_ROOT/global_r$((rr-1))"
  if [ -d "$GD" ] && [ -f "$GD/trunk.pth" ] && [ ! -f "$GD/extras.pth" ]; then
    echo "[WARN] $GD 缺 extras.pth（该轮聚合是老版本产物）—— 模态 token 将随机初始化"
  fi
done

for ((r=START_ROUND; r<NUM_ROUNDS; r++)); do
  echo "[round $r] 开始"

  # 全局目录（本轮聚合产物）
  GLOBAL_DIR="$OUTPUT_ROOT/global_r${r}"
  mkdir -p "$GLOBAL_DIR"

  # 上一轮全局目录（第 0 轮用 INIT_PRETRAINED）
  if [ "$r" -eq 0 ]; then
    PRETRAINED="$INIT_PRETRAINED"
  else
    PRETRAINED="$OUTPUT_ROOT/global_r$((r-1))"
  fi

  # ---- 客户端串行本地训练 ----
  for ((c=0; c<NUM_CLIENTS; c++)); do
    DOMAINS="${CLIENT_DOMAINS[$c]}"
    CLIENT_ENV="${CLIENT_ENVS[$c]}"
    # 逐客户端的步数：epoch_size 随任务 episode 长度不同而不同，
    # 必须按任务分别算 total_iters（单位是 batch step，不是 epoch）。
    C_EPOCH_SIZE="${CLIENT_EPOCH_SIZE[$c]}"
    C_TOTAL_ITERS=$((LOCAL_EPOCHS * C_EPOCH_SIZE - 1))
    C_WARMUP=$(( (C_TOTAL_ITERS + WARMUP_RATIO_NUM - 1) / WARMUP_RATIO_NUM ))
    # 实际执行的优化步数 = LOCAL_EPOCHS × epoch_size
    # （total_iters 是"停止阈值"，真实步数比它多 1，见 fedavg_config.sh 的说明）
    C_LOCAL_STEPS=$((LOCAL_EPOCHS * C_EPOCH_SIZE))
    CLIENT_OUT="$OUTPUT_ROOT/client_${c}_r${r}"
    LOG="$OUTPUT_ROOT/log_client_${c}_r${r}.txt"
    echo "[round $r] client $c  domains=$DOMAINS env=$CLIENT_ENV  epoch_size=$C_EPOCH_SIZE" \
         "total_iters=$C_TOTAL_ITERS warmup=$C_WARMUP -> $CLIENT_OUT"

    # ---- 全局调度的进度区间 ----
    # 第 r 轮对应全局进度 [r/R, (r+1)/R]，轮内再按本地步数细分。
    # 这样 50 轮的 lr 轨迹与同预算集中式逐点相同。
    if [ "$USE_GLOBAL_LR" = "1" ]; then
      P_START=$(awk "BEGIN{printf \"%.10f\", $r / $NUM_ROUNDS}")
      P_END=$(awk "BEGIN{printf \"%.10f\", ($r + 1) / $NUM_ROUNDS}")
      export FEDVLA_GLOBAL_LR=1
      export FEDVLA_P_START="$P_START"
      export FEDVLA_P_END="$P_END"
      export FEDVLA_NUM_STEPS="$C_LOCAL_STEPS"
      export FEDVLA_WARMUP_FRAC="$WARMUP_FRAC"
      # Adam 状态跨轮持久化（每个客户端一份；同一客户端的域固定，参数量稳定）
      # FEDVLA_PERSIST_OPT=0 可关掉（用于消融：只开全局 lr，不持久化 Adam）
      if [ "${FEDVLA_PERSIST_OPT:-1}" = "1" ]; then
        export FEDVLA_OPT_STATE="$OUTPUT_ROOT/opt_state_c${c}.pt"
      else
        unset FEDVLA_OPT_STATE
      fi
    fi

    # ---- 专家计数导出路径（EDA 的输入信号）----
    if [ "$USE_MOE" = "1" ]; then
      export FEDVLA_MOE_COUNTS="$CLIENT_OUT/expert_counts.pt"
    fi

    "$HPT_PY" -m hpt.run \
      env=mujoco_metaworld \
      train.pretrained_dir="$PRETRAINED" \
      dataset.episode_cnt="$EPISODE_CNT" \
      domains="$DOMAINS" \
      +datasetsize="$DATASETSIZE" \
      "dataset_generator_func.env_names=[\"$CLIENT_ENV\"]" \
      "rollout_runner.env_names=[\"$CLIENT_ENV\"]" \
      "rollout_runner.episode_num=$TRAIN_EVAL_ROLLOUTS" \
      train.total_iters="$C_TOTAL_ITERS" \
      output_dir="$CLIENT_OUT" \
      train.freeze_trunk=False \
      warmup_lr.step="$C_WARMUP" \
      > "$LOG" 2>&1

    if [ ! -f "$CLIENT_OUT/model.pth" ]; then
      echo "[round $r] client $c 训练失败，查看 $LOG"; tail -n 30 "$LOG"; exit 1
    fi
    # 记录本客户端实际的 epoch_size 与步数（首轮实测优先于配置值）
    grep -m1 "Epoch size:" "$LOG" | sed "s/^/[round $r] client $c /" || true
  done

  # ---- 续训正确性自检（只对第 1 轮及以后做）----
  # 目的：证明客户端模型里的 trunk 与上一轮全局模型**确实不同**，
  #      即聚合的权重真的被载入并继续训练了。
  #      若相同，说明续训没生效（load_trunk 空操作 / 权重没被覆盖）→ 立刻停。
  if [ "$r" -ge 1 ]; then
    PREV_GLOBAL="$OUTPUT_ROOT/global_r$((r-1))"
    if [ -f "$PREV_GLOBAL/trunk.pth" ]; then
      "$HPT_PY" - "$PREV_GLOBAL/trunk.pth" "$CLIENT_OUT/model.pth" <<'PY' || exit 1
import sys, torch
prev_path, cur_path = sys.argv[1], sys.argv[2]
prev = torch.load(prev_path, map_location="cpu", weights_only=False)
cur = torch.load(cur_path, map_location="cpu", weights_only=False)
# 全局 trunk.pth 的键是 'trunk.xxx'；完整模型里对应 'trunk.trunk.xxx'
diffs = []
for k, v in prev.items():
    ck = "trunk." + k
    if ck in cur:
        diffs.append((float((cur[ck].float() - v.float()).abs().max()), k))
if not diffs:
    print("[!!] 无法比对 trunk 键（键名不匹配）")
    sys.exit(1)
mx = max(diffs)
print(f"[续训自检] 客户端 trunk 与上一轮全局的最大差异 = {mx[0]:.3e} @ {mx[1]}")
if mx[0] == 0.0:
    print("[!!] 客户端 trunk 与上一轮全局完全相同 —— 聚合权重未被载入，联邦是空转！")
    sys.exit(1)
print("[OK] 续训生效：客户端确实在上一轮全局权重基础上继续训练")
PY
    fi
  fi

  # ---- 服务器端聚合 ----
  CLIENT_DIRS=()
  for ((c=0; c<NUM_CLIENTS; c++)); do CLIENT_DIRS+=("$OUTPUT_ROOT/client_${c}_r${r}"); done
  AGG_ARGS=( --clients "${CLIENT_DIRS[@]}" --out "$GLOBAL_DIR" )
  if [ "$AGG_MODE" = "eda" ] || [ "$AGG_MODE" = "eda_cluster" ]; then
    # EDA：逐层权重由各客户端的专家计数算出（无需外部指定 --weights）
    TASKS=()
    for ((c=0; c<NUM_CLIENTS; c++)); do TASKS+=("${CLIENT_DOMAINS[$c]}"); done
    AGG_ARGS+=( --eda --tasks "${TASKS[@]}" )
    if [ "$AGG_MODE" = "eda_cluster" ]; then
      AGG_ARGS+=( --eda-cluster )
      if [ -n "${EDA_NUM_CLUSTERS:-}" ]; then
        AGG_ARGS+=( --num-clusters "$EDA_NUM_CLUSTERS" )
      fi
    fi
    echo "[round $r] 聚合模式: $AGG_MODE（tasks: ${TASKS[*]}）"
  elif [ "$FEDAVG_WEIGHTS" = "data" ]; then
    # 教科书 FedAvg：按各客户端本地样本数(=epoch_size)加权
    W=()
    for ((c=0; c<NUM_CLIENTS; c++)); do W+=("${CLIENT_EPOCH_SIZE[$c]}"); done
    AGG_ARGS+=( --weights "${W[@]}" )
    echo "[round $r] 聚合权重(按样本数): ${W[*]}"
  else
    echo "[round $r] 聚合权重: 等权"
  fi
  "$HPT_PY" "$SCRIPT_DIR/aggregate.py" "${AGG_ARGS[@]}" \
    >> "$OUTPUT_ROOT/log_aggregate_r${r}.txt" 2>&1

  if [ ! -f "$GLOBAL_DIR/trunk.pth" ]; then
    echo "[round $r] 聚合失败，查看 $OUTPUT_ROOT/log_aggregate_r${r}.txt"; exit 1
  fi
  echo "[round $r] 完成，全局模型 -> $GLOBAL_DIR"

  # ---- 磁盘水位守卫 + 清理历史轮次权重 ----
  FG=$(free_gb)
  KEEP=$KEEP_ROUNDS
  if [ "$FG" -lt "$MIN_FREE_GB" ]; then
    echo "[!!] 可用空间仅 ${FG}GB < ${MIN_FREE_GB}GB —— 强制只保留最近 1 轮"
    KEEP=1
  fi
  cleanup_old_rounds "$r" "$KEEP"
done

echo "===== 全部 $NUM_ROUNDS 轮完成 ====="
echo "最终全局模型: $OUTPUT_ROOT/global_r$((NUM_ROUNDS-1))"
echo "下一步：用 federated/run_eval_fedavg.sh 评估（3 seeds × 20 rollouts）"
