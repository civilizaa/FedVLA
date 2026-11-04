#!/usr/bin/env bash
# FedVLA 门2'：把「联邦管线损耗」拆开的修复 + 消融实验链（幂等，可反复执行）。
#
# 背景
# ----
# 门2 测出三层效应：训练量 +0.050 → 联邦管线损耗 −0.317 → 异构专有损害 −0.616。
# 但"联邦管线损耗"的**归因**还没做。候选机制有三个：
#   (a) lr 轨迹是锯齿波：hpt/run.py 把 cosine 的 T_max 硬编码为 train.total_iters，
#       联邦下 = 单轮本地步数(44/56/71)；加上每轮新进程 → 50 次 "warmup→cosine→0"
#   (b) Adam 动量每轮清零：opt 每轮在进程内重建
#   (c) 模型每轮被平均（IID 下应恒等，异构下不是）
#
# 实测（真实 torch 调度器逐字复刻）已排除"学习量积分塌了"这个猜测：
#   联邦原生学习量∫ 1449.4  vs  集中式原生 1438.1 = 100.8%（等效步数 1500 vs 1487）
# 所以要靠消融回答 (a)/(b)/(c) 各自占多少。
#
# 本脚本按优先级跑 5 个实验，每个都可跳过（产物存在即跳过）：
#   V1  IID 联邦 10 轮，只开全局 lr                    （隔离 a）
#   V2  IID 联邦 10 轮，全局 lr + Adam 持久化           （隔离 a+b）★ 与 V1 对比
#   E3  集中式等预算 task1，全局 lr                      （同调度口径的上界）
#   E4  IID 联邦 50 轮，全局 lr + Adam 持久化            （修复后的联邦基线）★
#   E5  异构联邦 50 轮，全局 lr + Adam 持久化            （第 4 章正式结论）★
#
# 关键判据：E4 应当逼近 E3 —— 因为 IID 下 4 个客户端数据/种子相同，
# 参数平均是恒等变换，IID 联邦 ≡ 单客户端分 50 段训练。
# 若 E4 仍远低于 E3，说明差距来自别处，需要继续查（见 §4.4）。
#
# 用法：
#   bash run_gate2b.sh                # 跑完整链
#   STAGES="V1 V2" bash run_gate2b.sh # 只跑指定阶段
#   STAGES="E4 E5" bash run_gate2b.sh
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPT_PY=/root/miniconda3/envs/hpt/bin/python
OUT=/root/HPT/output
STAGES=${STAGES:-"V1 V2 E3 E4 E5"}
LOGDIR="$OUT/g2b"
mkdir -p "$LOGDIR"

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# 异构划分（默认）：c0=reach c1=push c2=button c3=reach
# IID 划分：4 个客户端全是 task1
IID_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1"
IID_ENVS="reach-v2 reach-v2 reach-v2 reach-v2"
IID_ESIZE="15 15 15 15"

has_stage () { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }
banner () { echo; echo "=============================================================="; echo ">>> $*"; echo "=============================================================="; }

# ── 阶段 0：打补丁（幂等）────────────────────────────────────────────────
banner "阶段 0：打全局步数感知调度补丁"
if "$HPT_PY" "$SCRIPT_DIR/patch_hpt_global_lr.py"; then
  echo "[OK] 补丁就绪"
else
  echo "[!!] 补丁失败，中止"; exit 1
fi
"$HPT_PY" "$SCRIPT_DIR/patch_hpt_global_lr.py" --verify || exit 1
# 顺便确认 load_trunk 补丁还在（联邦续训的前提）
if ! grep -q "FedVLA patch" /root/HPT/hpt/models/policy.py; then
  echo "[!!] load_trunk 补丁缺失！请先执行 patch_hpt_load_trunk.py"; exit 1
fi
echo "[OK] load_trunk 补丁在位"

# ── V1 / V2 / E4：IID 联邦 ───────────────────────────────────────────────
run_iid () {   # $1=tag $2=rounds $3=是否持久化 Adam
  local tag="$1" rounds="$2" optstate="$3"
  local root="$OUT/$tag"
  if [ -f "$root/global_r$((rounds-1))/trunk.pth" ]; then
    echo "[SKIP] $tag 已完成（$root/global_r$((rounds-1))/trunk.pth 存在）"
    return 0
  fi
  banner "IID 联邦：$tag  rounds=$rounds  Adam持久化=$optstate"
  ( cd /root/HPT && \
    OUTPUT_ROOT="$root" \
    NUM_ROUNDS="$rounds" \
    USE_GLOBAL_LR=1 \
    WARMUP_FRAC=0.25 \
    CLIENT_DOMAINS_OVERRIDE="$IID_DOMAINS" \
    CLIENT_ENVS_OVERRIDE="$IID_ENVS" \
    CLIENT_EPOCH_SIZE_OVERRIDE="$IID_ESIZE" \
    FEDVLA_PERSIST_OPT="$optstate" \
    bash "$SCRIPT_DIR/run_fedavg.sh" ) > "$LOGDIR/${tag}.log" 2>&1
  local rc=$?
  echo "[---] $tag rc=$rc，日志尾部："; tail -n 15 "$LOGDIR/${tag}.log"
  return $rc
}

# run_fedavg.sh 里的 Adam 持久化开关：FEDVLA_PERSIST_OPT=0/1
# （run_fedavg.sh 用它决定是否 export FEDVLA_OPT_STATE）

if has_stage V1; then
  run_iid g2b_iid_v1 10 0 || echo "[WARN] V1 失败，继续下一阶段"
fi
if has_stage V2; then
  run_iid g2b_iid_v2 10 1 || echo "[WARN] V2 失败，继续下一阶段"
fi
if has_stage E4; then
  run_iid g2b_iid_e4 50 1 || echo "[WARN] E4 失败，继续下一阶段"
fi

# ── E3：集中式等预算（task1），同调度口径 ────────────────────────────────
if has_stage E3; then
  root="$OUT/g2b_central_eq"
  if [ -f "$root/task1/model.pth" ]; then
    echo "[SKIP] E3 已完成"
  else
    banner "E3：集中式等预算 task1（2250 步，全局调度）"
    ( cd /root/HPT && \
      OUT_ROOT="$root" ONLY_TASK=task1 ITERS=2250 USE_GLOBAL_LR=1 WARMUP_FRAC=0.25 \
      bash "$SCRIPT_DIR/run_centralized_equal_budget.sh" ) > "$LOGDIR/e3_central_eq.log" 2>&1
    echo "[---] E3 rc=$?，日志尾部："; tail -n 15 "$LOGDIR/e3_central_eq.log"
  fi
fi

# ── E5：异构联邦 50 轮（正式结论）────────────────────────────────────────
if has_stage E5; then
  root="$OUT/g2b_hetero"
  if [ -f "$root/global_r49/trunk.pth" ]; then
    echo "[SKIP] E5 已完成"
  else
    banner "E5：异构联邦 50 轮（全局 lr + Adam 持久化）"
    ( cd /root/HPT && \
      OUTPUT_ROOT="$root" NUM_ROUNDS=50 USE_GLOBAL_LR=1 WARMUP_FRAC=0.25 \
      FEDVLA_PERSIST_OPT=1 \
      bash "$SCRIPT_DIR/run_fedavg.sh" ) > "$LOGDIR/e5_hetero.log" 2>&1
    echo "[---] E5 rc=$?，日志尾部："; tail -n 15 "$LOGDIR/e5_hetero.log"
  fi
fi

# ── 汇总：把各阶段的收敛曲线抽出来对比 ───────────────────────────────────
banner "收敛曲线对比（各阶段末轮 / 指定轮次的联邦平均 loss）"
for d in g2b_iid_v1 g2b_iid_v2 g2b_iid_e4 g2b_hetero; do
  if [ -d "$OUT/$d" ]; then
    echo "--- $d ---"
    "$HPT_PY" "$SCRIPT_DIR/extract_convergence.py" --root "$OUT/$d" \
      --out "$OUT/$d/conv.json" 2>&1 | grep -E '^ *[0-9]+ \|' | tail -12
  fi
done

echo
echo "GATE2B_DONE"
