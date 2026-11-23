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

# ===== HPT 源码必须在 import 路径上（2026-11-18 血案）=====
# `python -m hpt.run` 靠 sys.path 找 hpt 包：HPT 在云端**没有** pip install（只有源码树），
# 所以只有 CWD 在 /root/HPT 时才能 import。
# DP 标定脚本是从 /root/fedcore 启动的 ⇒ CWD=/root ⇒ 报
#   "No module named 'hpt'" → 标定直接失败 → DP 阶段整条链断掉（实测踩坑）。
# 正解：显式把 HPT 源码目录挂进 PYTHONPATH，与 CWD 无关。
export HPT_ROOT="${HPT_DIR:-/root/HPT}"
# 🔴 钩子模块在 /root/HPT/hpt/utils/ 下，要 import 同伴的 fed_baselines.py
#    必须告诉它基座目录（云端是 /root/fedcore，本地是 <repo>/federated）。
export FEDVLA_FED_BASE="${FEDVLA_FED_BASE:-$SCRIPT_DIR}"
export PYTHONPATH="$HPT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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

# ===== _baseline 历史轮次清理（2026-11-23 修复）=================================
# 背景：FedProx 的 anchor_r{r}.pt（84MB）与 SCAFFOLD 的 ci_c{c}_r{r}.pt（50MB）
#       原本**从不被清理** ⇒ 单配置 4.2G / 10.1G ⇒ 30G 系统盘必爆。
#       2026-11-23 实测：p5_bal_scaffold 在第 1 轮就因磁盘水位触发
#       「可用空间仅 5GB < 6GB」而聚合中止。
# 口径：只删**已被消费完**的中间产物，保留最近 $KEEP 轮；
#       ci_c{c}.pt（当前控制变量）与 c_global_r{r}.pt（SCAFFOLD 下发源）永不删。
cleanup_old_baseline () {
  local cur="$1" keep="$2"
  local cutoff=$((cur - keep))
  [ "$cutoff" -lt 1 ] && return 0
  local n=0 f base r
  for f in "$BASELINE_DIR"/anchor_r*.pt "$BASELINE_DIR"/ci_c*_r*.pt \
           "$BASELINE_DIR"/c_global_r*.pt "$BASELINE_DIR"/wglobal_r*.pt; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    case "$base" in
      ci_c[0-9].pt) continue ;;   # 当前控制变量的持久化副本（无轮次号，永不删）
    esac
    # 🔴 必须**按轮次**判断，只删 r < cutoff 的；
    #    原实现不判轮次 ⇒ 会把**当前轮**的 anchor_r{r}.pt / ci_c{c}_r{r}.pt 一起删掉
    #    ⇒ SCAFFOLD 的 c_k 下发链断裂 ⇒ 下一轮聚合中止。
    r=$(echo "$base" | grep -oE '_r[0-9]+\.pt$' | grep -oE '[0-9]+')
    [ -n "$r" ] || continue
    if [ "$r" -lt "$cutoff" ]; then
      rm -f "$f" && n=$((n + 1))
    fi
  done
  [ "$n" -gt 0 ] && echo "[清理] 删除 $n 个 _baseline 历史轮次中间产物（保留最近 ${keep} 轮）"
  return 0
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

# ===== 开训前硬拦截：HPT 源码必须可 import（2026-11-18 血案）=====
# `python -m hpt.run` 在云端只能靠 PYTHONPATH 找到 hpt 包（HPT 未 pip install）。
# 之前 DP 标定从 /root/fedcore 启动 ⇒ "No module named 'hpt'" ⇒ 标定失败 ⇒
# DP 阶段整条链断掉，而**错误信息藏在日志里**、上层只看到 "标定失败(rc=1)"。
# 这里提前硬拦截：import 不通就直接退出，并把 PYTHONPATH 打出来。
if ! "$HPT_PY" -c "import hpt" >/dev/null 2>&1; then
  echo "[!!] python -c 'import hpt' 失败 —— HPT 源码不在 import 路径上！"
  echo "     HPT_ROOT=$HPT_ROOT"
  echo "     PYTHONPATH=${PYTHONPATH:-<空>}"
  echo "     CWD=$(pwd)"
  echo "     修法：export PYTHONPATH=\"$HPT_ROOT:\$PYTHONPATH\"，或 cd 到 \$HPT_ROOT 后再跑"
  exit 1
fi
echo "[OK] hpt 可 import（PYTHONPATH=$HPT_ROOT）"

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

# ===== 第 5 阶段：联邦基线（FedProx / SCAFFOLD）=======================
# 路线图 6.1 主实验矩阵要求与 FedProx、SCAFFOLD 对比 —— 这两个是通信评委
# 最熟悉的联邦基线，此前全仓未实现（2026-11-22 补齐）。
#
# BASELINE 三态：
#   ""        → 完全关闭（默认）。此时本段代码整块跳过，
#               训练行为与改造前**逐字节一致** ⇒ 不影响已完成的 1~4 阶段实验。
#   fedprox   → 近端项 μ/2·‖w − w_global‖²，梯度修正注入 backward 之后
#   scaffold  → 控制变量修正 g ← g − c_k + c（客户端漂移校正）
#
# ⚠️ 这两个方法**不改服务器的参数聚合权重**，只改**客户端本地更新方向**，
#    所以 AGG_MODE 仍可为 fedavg/eda/eda_cluster（可与 MoE 组合）。
BASELINE=${BASELINE:-}
PROX_MU=${PROX_MU:-0.01}          # FedProx 近端强度（论文常用 0.01~1）
SCAFFOLD_LR=${SCAFFOLD_LR:-}      # 本地平均 lr；留空则由训练侧统计

if [ -n "$BASELINE" ]; then
  if [ "$BASELINE" != "fedprox" ] && [ "$BASELINE" != "scaffold" ]; then
    echo "[!!] BASELINE=$BASELINE 非法（只能是 fedprox / scaffold / 空）"
    exit 1
  fi
  # 硬拦截：HPT 的基线补丁必须已打上，否则会**静默退化成 FedAvg**
  # （这正是第 4 阶段踩过的"silent failure"套路：没报错，结果全废）。
  if ! grep -q "apply_baseline_hook" "${HPT_ROOT}/hpt/train_test.py" 2>/dev/null; then
    echo "[!!] HPT 未打联邦基线补丁（hpt/train_test.py 里没有 apply_baseline_hook）"
    echo "    先执行： python $SCRIPT_DIR/patch_hpt_fedbaseline.py --hpt-dir $HPT_ROOT"
    exit 1
  fi
  echo "[OK] 联邦基线 = $BASELINE  (PROX_MU=$PROX_MU)"
fi

# SCAFFOLD 的控制变量目录（逐客户端持久化 c_k + 服务器全局 c）
BASELINE_DIR="${BASELINE_DIR:-$OUTPUT_ROOT/_baseline}"
mkdir -p "$BASELINE_DIR"

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

    # ---- 第 5 阶段：联邦基线环境变量（FedProx / SCAFFOLD）----
    # ⚠️ BASELINE 为空时，本段整块跳过 ⇒ 训练行为与改造前逐字节一致。
    #    FedProx 需要 w_global（= 本轮起始的全局模型）：
    #      r=0 时是 INIT_PRETRAINED（hf 预训练），r>0 时是上一轮的 global_r{r-1}。
    #      客户端已从它续训，所以把它显式落盘给钩子读取（钩子只认 .pt）。
    if [ -n "$BASELINE" ]; then
      export FEDVLA_BASELINE="$BASELINE"
      # ---- 锚点 / w_global 的落盘（键空间必须是**全模型键空间**）----
      # ⚠️ 键空间陷阱（2026-11-22）：模型参数名是 `trunk.trunk.xxx`，
      #    而聚合写出的 trunk.pth 的键是 `trunk.xxx`（剥掉了第一层前缀）。
      #    直接拿 trunk.pth 当锚点 ⇒ 键对不上 ⇒ 修正项作用不到任何参数
      #    ⇒ 基线**静默**退化成 FedAvg（零报错）。
      #    ⇒ 这里统一把前缀补回去，变成 `trunk.trunk.xxx`。
      _write_anchor () {   # $1=源目录或 hf://   $2=输出 .pt
        "$HPT_PY" - "$1" "$2" <<'PYANCHOR'
import os, sys, torch
src, dst = sys.argv[1], sys.argv[2]
if src.startswith("hf://"):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("liruiw/hpt-base", "trunk.pth", repo_type="model")
    sd = torch.load(p, map_location="cpu", weights_only=False)
else:
    tp = os.path.join(src, "trunk.pth")
    if not os.path.exists(tp):
        raise SystemExit(f"[FATAL] 找不到 {tp}")
    sd = torch.load(tp, map_location="cpu", weights_only=False)
# 键空间转换：trunk.pth 的 `trunk.xxx` -> 全模型键空间的 `trunk.trunk.xxx`
# ⚠️ trunk.pth 的键**本来就以 "trunk." 开头**（那是 Policy.trunk 的内部键），
#    所以这里必须**无条件**加前缀；写成"若不以 trunk. 开头才加"会漏转换
#    ⇒ 键对不上 ⇒ 基线静默退化成 FedAvg（2026-11-22 单测抓到）。
if sd and all(k.startswith("trunk.trunk.") for k in sd):
    full = sd                      # 已经是全模型键空间，不动
else:
    full = {"trunk." + k: v for k, v in sd.items()}
torch.save(full, dst)
print(f"[baseline] 锚点已落盘: {dst} 键数={len(full)}（全模型键空间）")
PYANCHOR
      }
      if [ "$BASELINE" = "fedprox" ]; then
        export FEDVLA_PROX_MU="$PROX_MU"
        ANCHOR_PT="$BASELINE_DIR/anchor_r${r}.pt"
        _write_anchor "$PRETRAINED" "$ANCHOR_PT" || exit 1
        export FEDVLA_PROX_ANCHOR="$ANCHOR_PT"
      elif [ "$BASELINE" = "scaffold" ]; then
        # c：服务器下发的全局控制变量（第 0 轮无 => 钩子走零初始化）
        CG="$BASELINE_DIR/c_global_r${r}.pt"
        if [ -f "$CG" ]; then export FEDVLA_SCAFFOLD_C="$CG"; else unset FEDVLA_SCAFFOLD_C; fi
        # c_k：本客户端的本地控制变量（**逐轮持久化**）
        #    ⚠️ 若不持久化，SCAFFOLD 会退化成 no-op（g - c_k + c = g），零报错。
        CL="$BASELINE_DIR/ci_c${c}.pt"
        if [ -f "$CL" ]; then export FEDVLA_SCAFFOLD_CV_LOCAL="$CL"; else unset FEDVLA_SCAFFOLD_CV_LOCAL; fi
        # c_k 更新需要的 w_global（本轮起始全局模型）与本地步数 K
        WG="$BASELINE_DIR/wglobal_r${r}.pt"
        _write_anchor "$PRETRAINED" "$WG" || exit 1
        export FEDVLA_SCAFFOLD_WGLOBAL="$WG"
        export FEDVLA_SCAFFOLD_CV_OUT="$BASELINE_DIR/ci_c${c}_r${r}.pt"
        export FEDVLA_SCAFFOLD_STEPS="$C_LOCAL_STEPS"
        if [ -z "$SCAFFOLD_LR" ]; then
          # 本地平均 lr：从训练日志里取最后一次的 lr（调度器输出）
          LR_EST=$(grep -oE "lr[^0-9.eE+-]*[0-9.eE+-]+" "$LOG" 2>/dev/null | tail -1 \
                   | grep -oE "[0-9.eE+-]+$" || true)
          export FEDVLA_SCAFFOLD_LR="${LR_EST:-1e-4}"
        else
          export FEDVLA_SCAFFOLD_LR="$SCAFFOLD_LR"
        fi
      fi
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

    # ---- 上行 DP 加噪钩子（第4阶段 DP-效用扫描；**默认关闭**）----
    # ⚠️ 默认 POST_CLIENT_HOOK 为空 ⇒ 这段代码整块跳过，
    #    训练行为与改造前**逐字节一致**，对任何既有实验零影响。
    # 用途：DP-FedAvg 形式的上行隐私 —— 客户端在上传共享参数前，
    #    对更新做 L2 裁剪 + 高斯加噪（Δ·min(1,C/‖Δ‖) + N(0,(σC)²)）。
    #    ε 只依赖 σ（噪声倍率），C 只影响效用 ⇒ C 可按实测更新范数标定。
    # 与隐私侧 dp.py 同口径：上行 = 非 stems./heads. 的全部键。
    if [ -n "${POST_CLIENT_HOOK:-}" ]; then
      HOOK_ARGS=(
        --client-dir "$CLIENT_OUT"
        --prev-global "$PRETRAINED"
        --sigma "${DP_SIGMA:-0}"
        --clip-norm "${DP_CLIP_NORM:-1.0}"
        --seed "${DP_SEED:-1234}"
        --round "$r" --client "$c"
      )
      [ "${DP_CALIBRATE:-0}" = "1" ] && HOOK_ARGS+=( --calibrate )
      [ -n "${DP_STAT_OUT:-}" ] && HOOK_ARGS+=( --stat-out "$DP_STAT_OUT" )
      "$HPT_PY" "$SCRIPT_DIR/$POST_CLIENT_HOOK" "${HOOK_ARGS[@]}" \
        >> "$OUTPUT_ROOT/log_dp_noise.txt" 2>&1 \
        || { echo "[round $r] client $c DP 钩子失败，查看 $OUTPUT_ROOT/log_dp_noise.txt";
             tail -n 20 "$OUTPUT_ROOT/log_dp_noise.txt"; exit 1; }
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

  # ---- 上行通信损伤注入（第4阶段，可选，默认关闭）----
  # COMM_DROP_RATE  ：本轮上行丢失的客户端比例（0=不丢）。用"固定丢 k 个 + 轮换"口径，
  #                   不用 Bernoulli —— 实测 Bernoulli 在 n=4、rate=0.1 时会连续 50 轮
  #                   一个都不丢（损伤被随机性完全稀释），做不出曲线。
  # COMM_QUANT_SCHEME：上行共享参数的量化方案（fp32/int8_g/int4_g/bit1/...）。
  # ⚠️ 第 0 轮**不注入丢包**：初始轮需要所有客户端在场，否则被丢弃客户端独占的域
  #    在全局模型里缺失，后续轮次从这个全局模型续训也会一直缺失 → 评估直接崩。
  CHANNEL_ARGS=()
  if [ -n "${COMM_DROP_RATE:-}" ] && [ "${COMM_DROP_RATE}" != "0" ] && [ "$r" -ge 1 ]; then
    DROPPED_JSON=$("$HPT_PY" "$SCRIPT_DIR/comm_channel.py" drop \
        --n "$NUM_CLIENTS" --rate "$COMM_DROP_RATE" \
        --seed "${COMM_SEED:-1}" --round "$r" --mode "${COMM_DROP_MODE:-fixed}")
    DROPPED_CSV=$(printf '%s' "$DROPPED_JSON" | tr -d '[] ')
    if [ -n "$DROPPED_CSV" ]; then
      # ⚠️ 2026-11-08 修正：argparse 的 nargs="*" 要求**空格分隔**（--drop-clients 0 1）。
      #    传逗号串 "0,1" 会报 "invalid int value: '0,1'" ⇒ 该轮聚合直接失败。
      #    c_drop25（只丢 1 个）侥幸通过，c_drop50/c_drop75 在 round 1 就静默失败。
      DROPPED_ARGS=$(printf '%s' "$DROPPED_CSV" | tr ',' ' ')
      CHANNEL_ARGS+=( --drop-clients $DROPPED_ARGS )
      CHANNEL_ARGS+=( --prev-global "$OUTPUT_ROOT/global_r$((r-1))" )
      echo "[round $r] 通信损伤：本轮丢弃客户端 ($DROPPED_CSV)，回退源 global_r$((r-1))"
    else
      echo "[round $r] 通信损伤：本轮无客户端被丢弃"
    fi
  fi
  if [ -n "${COMM_QUANT_SCHEME:-}" ] && [ "$COMM_QUANT_SCHEME" != "fp32" ]; then
    CHANNEL_ARGS+=( --quant-scheme "$COMM_QUANT_SCHEME" )
    echo "[round $r] 通信损伤：上行量化方案 = $COMM_QUANT_SCHEME"
  fi

  # ---- 服务器端聚合 ----
  CLIENT_DIRS=()
  for ((c=0; c<NUM_CLIENTS; c++)); do CLIENT_DIRS+=("$OUTPUT_ROOT/client_${c}_r${r}"); done
  AGG_ARGS=( --clients "${CLIENT_DIRS[@]}" --out "$GLOBAL_DIR" )
  if [ "${#CHANNEL_ARGS[@]}" -gt 0 ]; then
    AGG_ARGS+=( "${CHANNEL_ARGS[@]}" )
  fi
  AGG_ARGS+=( --comm-report "$OUTPUT_ROOT/comm_report_r${r}.json" )
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

  # ---- 第 5 阶段：SCAFFOLD 服务端控制变量聚合 ----
  # 客户端在本地训练结束时已把更新后的本地控制变量写到
  # $BASELINE_DIR/ci_c{c}_r{r}.pt；这里聚合成全局 c，供下一轮下发。
  # ⚠️ 失败必须 exit 1（铁律 Q）：若这里失败却继续，下一轮客户端拿到的是
  #    **上一轮的旧 c** ⇒ SCAFFOLD 静默退化成近似 FedAvg ⇒ 结果全废且零报错。
  if [ "$BASELINE" = "scaffold" ]; then
    CV_DIRS=()
    for ((c=0; c<NUM_CLIENTS; c++)); do
      SRC="$BASELINE_DIR/ci_c${c}_r${r}.pt"
      if [ ! -f "$SRC" ]; then
        echo "[!!] 客户端 $c 缺本地控制变量 $SRC —— SCAFFOLD 聚合中止"
        exit 1
      fi
      # 客户端目录里放一份（aggregate-cv 从客户端目录读）
      cp "$SRC" "$OUTPUT_ROOT/client_${c}_r${r}/scaffold_cv.pt"
      CV_DIRS+=("$OUTPUT_ROOT/client_${c}_r${r}")
    done
    "$HPT_PY" "$SCRIPT_DIR/fed_baselines.py" aggregate-cv \
      --clients "${CV_DIRS[@]}" \
      --out "$BASELINE_DIR/c_global_r$((r+1)).pt" \
      >> "$OUTPUT_ROOT/log_baseline_r${r}.txt" 2>&1 \
      || { echo "[!!] SCAFFOLD 控制变量聚合失败，查看 $OUTPUT_ROOT/log_baseline_r${r}.txt";
           tail -n 20 "$OUTPUT_ROOT/log_baseline_r${r}.txt"; exit 1; }
    # 回写 c_k 的持久化副本（下一轮客户端要读它）
    # ⚠️ 顺序很重要：必须**先聚合成功**再回写。顺序反了的话，
    #    聚合失败时 c_k 已被覆盖 ⇒ 下一轮拿到错的 c_k ⇒ 静默失真。
    for ((c=0; c<NUM_CLIENTS; c++)); do
      cp "$BASELINE_DIR/ci_c${c}_r${r}.pt" "$BASELINE_DIR/ci_c${c}.pt"
    done
    echo "[round $r] SCAFFOLD 全局控制变量 -> $BASELINE_DIR/c_global_r$((r+1)).pt"
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
  [ -n "${BASELINE_DIR:-}" ] && cleanup_old_baseline "$r" "$KEEP"
done

echo "===== 全部 $NUM_ROUNDS 轮完成 ====="
echo "最终全局模型: $OUTPUT_ROOT/global_r$((NUM_ROUNDS-1))"
echo "下一步：用 federated/run_eval_fedavg.sh 评估（3 seeds × 20 rollouts）"
