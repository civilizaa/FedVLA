#!/usr/bin/env bash
# FedVLA 第 3 阶段：MoE（专家混合）+ EDA（专家驱动聚合）的实验链（幂等，可反复执行）。
#
# 背景
# ----
# 门2 测出三层效应：训练量 +0.050 → 联邦管线损耗 −0.317 → 异构专有损害 −0.616。
# 第 3 阶段要回答的是：**能不能把这个 −0.616 压下去？**
# 手段是 FedVLA 原文的两件武器：
#   1) MoE：让不同任务的数据路由到不同专家，在参数层面实现任务分化
#   2) EDA：用专家选择计数作为聚合信号，替代"人人都等权"
#
# 前置条件（**必须先跑完门2' 的 E5**）
# ----------------------------------
# 门2 的异构联邦基线（0.017）是在"每轮重启 LR + Adam 清零"的管线缺陷下测出来的，
# 天花板太低。如果直接在这个基线上加 MoE/EDA，效应会被管线损耗淹没，
# 分不清是 MoE/EDA 的功劳还是调度修复的功劳。
# 所以顺序必须是：门2'（E1~E5，修好调度）→ 本脚本。
#
# 实验矩阵
# --------
#   P3-A  异构 + MoE + FedAvg 等权        → p3_moe_fedavg      （MoE 单独的贡献）
#   P3-B  异构 + MoE + EDA 中心性         → p3_moe_eda         （中心性加权的贡献）
#   P3-C  异构 + MoE + EDA 簇感知         → p3_moe_eda_cluster ★H2「按簇聚合更优」
#   P3-D  IID  + MoE + FedAvg 等权        → p3_moe_iid         （对照：MoE 不伤 IID）
#
#   对照组（门2' 的产物，已在磁盘上）：
#     g2b_hetero      异构 + 修复调度 + FedAvg     ← P3-A/B/C 的基线
#     g2b_iid_e4      IID  + 修复调度 + FedAvg     ← P3-D 的基线
#     g2b_central_eq  集中式等预算                  ← 上界
#
# 判据
# ----
#   H2（P3-C vs g2b_hetero）：簇感知聚合应显著优于等权 FedAvg。
#      机理：异构划分下 reach-v2 占 2/4 的客户端，等权聚合让"多数任务"主导全局模型；
#      簇感知（簇间等权）把这个比例压到 1/3，直接削弱客户端漂移。
#   负结果也有价值：若 P3-C 不优于基线，说明"按簇聚合"在 VLA 上不成立，
#      可改写成"专家信号更适合做诊断而非聚合权重"的结论。
#
# 用法：
#   bash run_phase3.sh                    # 跑完整链
#   STAGES="P3-A P3-C" bash run_phase3.sh # 只跑指定阶段
#   FORCE=1 bash run_phase3.sh            # 忽略"已完成"跳过，重新跑
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPT_PY=/root/miniconda3/envs/hpt/bin/python
OUT=/root/HPT/output
STAGES=${STAGES:-"P3-A P3-B P3-C P3-D"}
LOGDIR="$OUT/p3"
FORCE=${FORCE:-0}
mkdir -p "$LOGDIR"

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# IID 划分（与门2' 一致）：4 个客户端全是 task1
IID_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1"
IID_ENVS="reach-v2 reach-v2 reach-v2 reach-v2"
IID_ESIZE="15 15 15 15"

# MoE 配置（与 fedvla_moe.py 的默认值一致）
MOE_EXPERTS=${MOE_EXPERTS:-8}
MOE_TOPK=${MOE_TOPK:-2}
MOE_EXPERT_RATIO=${MOE_EXPERT_RATIO:-0.25}
MOE_LAYERS=${MOE_LAYERS:-half}

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
banner () { echo; echo "=============================================================="; echo ">>> $*"; echo "=============================================================="; }

# ── 断点续跑：找出最后一个"完整"轮次，从 N+1 继续 ─────────────────────────
# 判据用 global_r<N>/trunk.pth —— 该文件只在**本轮 4 个客户端全部训练完 + 聚合完**
# 之后才生成，所以它存在 == 第 N 轮真正跑完（不是半截）。
# 安全性的三个依据：
#   1) 全局 lr 进度是 P_START = r / NUM_ROUNDS（按**绝对轮次**算，不是进程内步数）
#      → 从第 N+1 轮续跑的 lr 轨迹与从头跑到第 N+1 轮**逐点相同**
#   2) Adam 状态 opt_state_c*.pt 是每轮结束时覆写的
#      → global_r<N> 存在时，opt_state 正好是第 N 轮的状态，与续跑点匹配
#   3) run_fedavg.sh 的 cleanup_old_rounds(cur, keep) 只删 d < cur-keep 的轮次
#      → 续跑点 global_rN 不会被清理掉（PRETRAINED 依赖它）
detect_start_round () {
  local root="$1" last="$2" d
  for ((d=last; d>=0; d--)); do
    if [ -f "$root/global_r${d}/trunk.pth" ]; then
      echo $((d + 1)); return 0
    fi
  done
  echo 0
}

# ── 阶段 0：前置检查 + 打补丁 ─────────────────────────────────────────────
banner "阶段 0：前置检查 + 打 MoE 补丁"

# 0.1 门2' 必须跑完（否则基线不公平）
if [ "$FORCE" != "1" ] && [ ! -f "$OUT/g2b_hetero/global_r49/trunk.pth" ]; then
  echo "[!!] 门2' 的 E5（异构联邦 50 轮，修复调度）产物不存在："
  echo "     $OUT/g2b_hetero/global_r49/trunk.pth"
  echo "     必须先跑完 bash run_gate2b.sh —— 否则基线带着管线缺陷，实验无意义。"
  exit 1
fi
echo "[OK] 门2' 基线在位（$OUT/g2b_hetero）"

# 0.2 load_trunk 补丁（联邦续训的前提）
if ! grep -q "FedVLA patch" /root/HPT/hpt/models/policy.py; then
  echo "[!!] load_trunk 补丁缺失！请先执行 patch_hpt_load_trunk.py"; exit 1
fi
echo "[OK] load_trunk 补丁在位"

# 0.3 全局 lr 调度补丁（门2' 已打过，这里确保还在）
if ! grep -q "GlobalStepLRScheduler" /root/HPT/hpt/run.py; then
  echo "[!!] 全局 lr 调度补丁缺失！请先执行 patch_hpt_global_lr.py"; exit 1
fi
echo "[OK] 全局 lr 调度补丁在位"

# 0.4 MoE 补丁（必须打在 load_trunk 之后）
if "$HPT_PY" "$SCRIPT_DIR/patch_hpt_moe.py"; then
  echo "[OK] MoE 补丁就绪"
else
  echo "[!!] MoE 补丁失败，中止"; exit 1
fi
"$HPT_PY" "$SCRIPT_DIR/patch_hpt_moe.py" --verify || exit 1

# ── 运行器 ────────────────────────────────────────────────────────────────
run_exp () {  # $1=tag $2=agg_mode $3=client_split(iid|hetero)
  local tag="$1" agg="$2" split="$3"
  local root="$OUT/$tag"
  local last="$((NUM_ROUNDS - 1))"
  if [ "$FORCE" != "1" ] && [ -f "$root/global_r${last}/trunk.pth" ]; then
    echo "[SKIP] $tag 已完成（$root/global_r${last}/trunk.pth 存在）"
    return 0
  fi
  banner "$tag  聚合=$agg  划分=$split  rounds=$NUM_ROUNDS"
  # 断点续跑：实例中途关机时不从头再来
  local sr; sr=$(detect_start_round "$root" "$last")
  if [ "$sr" -gt 0 ]; then
    echo "[续跑] $tag 已有完整轮次到 r$((sr - 1))，从第 $sr 轮继续（省下 $sr 轮）"
  else
    echo "[新跑] $tag 从第 0 轮开始"
  fi
  local env_extra=()
  if [ "$split" = "iid" ]; then
    env_extra=(
      "CLIENT_DOMAINS_OVERRIDE=$IID_DOMAINS"
      "CLIENT_ENVS_OVERRIDE=$IID_ENVS"
      "CLIENT_EPOCH_SIZE_OVERRIDE=$IID_ESIZE"
    )
  fi
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
        "${env_extra[@]}" \
        bash "$SCRIPT_DIR/run_fedavg.sh" ) > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  echo "[---] $tag rc=$rc，日志尾部："; tail -n 20 "$LOGDIR/${tag}.log"
  return $rc
}

NUM_ROUNDS=${NUM_ROUNDS:-50}

if has_stage P3-A; then
  run_exp p3_moe_fedavg fedavg hetero || echo "[WARN] P3-A 失败，继续下一阶段"
fi
if has_stage P3-B; then
  run_exp p3_moe_eda eda hetero || echo "[WARN] P3-B 失败，继续下一阶段"
fi
if has_stage P3-C; then
  run_exp p3_moe_eda_cluster eda_cluster hetero || echo "[WARN] P3-C 失败，继续下一阶段"
fi
if has_stage P3-D; then
  run_exp p3_moe_iid fedavg iid || echo "[WARN] P3-D 失败，继续下一阶段"
fi

# ── 汇总：EDA 权重报告（论文里直接引用）──────────────────────────────────
banner "EDA 权重报告（各阶段末轮）"
for d in p3_moe_eda p3_moe_eda_cluster; do
  if [ -f "$OUT/$d/global_r$((NUM_ROUNDS-1))/eda_report.txt" ]; then
    echo "--- $d ---"
    cat "$OUT/$d/global_r$((NUM_ROUNDS-1))/eda_report.txt"
    echo
  fi
done

# ── 汇总：收敛曲线 ────────────────────────────────────────────────────────
banner "收敛曲线对比"
for d in g2b_hetero p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster; do
  if [ -d "$OUT/$d" ]; then
    echo "--- $d ---"
    "$HPT_PY" "$SCRIPT_DIR/extract_convergence.py" --root "$OUT/$d" \
      --out "$OUT/$d/conv.json" 2>&1 | grep -E '^ *[0-9]+ \|' | tail -12
  fi
done

echo
echo "PHASE3_DONE"
