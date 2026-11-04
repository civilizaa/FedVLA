#!/usr/bin/env bash
# FedVLA：等预算集中式基线（论文对比表的第二列）
#
# 为什么要等预算基线：
#   集中式 6000 步（= HPT 官方 Meta-World 配方，400 epoch）是**上界参照**，
#   它的预算远大于联邦客户端（联邦 50 轮 = task1 客户端 2250 步 = 150 epoch）。
#   论文里如果只报 6000 步，审稿人会问"差距是不是纯粹因为联邦训练得少"。
#   所以再跑一组**等预算**集中式（同样 2250 步），把"训练量"这个混淆因素排除掉。
#
# 用法：
#   bash run_centralized_equal_budget.sh          # 三个任务，ITERS 按各自 epoch_size 算
#   ITERS=2250 bash run_centralized_equal_budget.sh   # 统一步数
set -u

SCRIPT_DIR=/root/HPT/federated
OUT_ROOT=${OUT_ROOT:-/root/HPT/output/centralized_equal_budget}
FED_ROUNDS=${FED_ROUNDS:-50}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-3}
HPT_PY=/root/miniconda3/envs/hpt/bin/python

# 全局步数感知调度（与联邦同一套调度定义，保证 lr 轨迹逐点可比）
USE_GLOBAL_LR=${USE_GLOBAL_LR:-1}
WARMUP_FRAC=${WARMUP_FRAC:-0.25}
if [ "$USE_GLOBAL_LR" = "1" ]; then
  SCHED_FILE="${HPT_DIR:-/root/HPT}/hpt/utils/fedvla_lr.py"
  if [ ! -f "$SCHED_FILE" ]; then
    echo "[!!] 找不到 $SCHED_FILE —— 全局调度补丁未打！"
    echo "     请先执行：python /root/HPT/federated/patch_hpt_global_lr.py"
    exit 1
  fi
  echo "[OK] 全局步数感知调度（warmup_frac=$WARMUP_FRAC）"
fi

export WANDB_MODE=offline MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# task -> (domain, env, epoch_size)
TASKS=(
  "task1 mujoco_metaworld_task1 reach-v2 15"
  "task2 mujoco_metaworld_task2 push-v2 19"
  "task3 mujoco_metaworld_task3 button-press-topdown-v2 24"
)

mkdir -p "$OUT_ROOT"

for t in "${TASKS[@]}"; do
  set -- $t
  task="$1"; dom="$2"; env="$3"; esize="$4"
  # ONLY_TASK=task1 可以只跑一个任务（消融实验用，省时间）
  if [ -n "${ONLY_TASK:-}" ] && [ "$task" != "$ONLY_TASK" ]; then
    echo "[SKIP] $task（ONLY_TASK=$ONLY_TASK）"
    continue
  fi

  # 每个客户端的等预算步数：轮数 × 本地 epoch × 该任务的 epoch_size
  if [ -n "${ITERS:-}" ]; then
    iters="$ITERS"
  else
    iters=$((FED_ROUNDS * LOCAL_EPOCHS * esize))
  fi
  # warmup 按本地总步数的 1/4（与联邦一致），保证学习率曲线可比
  warmup=$(( (iters + 3) / 4 ))

  out="$OUT_ROOT/$task"
  log="$OUT_ROOT/train_$task.log"
  if [ -f "$out/model.pth" ]; then
    echo "[SKIP] $task 已训好：$out/model.pth"
    continue
  fi

  echo "===== 等预算集中式 $task  domain=$dom env=$env  iters=$iters warmup=$warmup ====="
  t0=$(date +%s)
  if [ "$USE_GLOBAL_LR" = "1" ]; then
    export FEDVLA_GLOBAL_LR=1
    export FEDVLA_P_START=0
    export FEDVLA_P_END=1
    export FEDVLA_NUM_STEPS="$iters"
    export FEDVLA_WARMUP_FRAC="$WARMUP_FRAC"
    unset FEDVLA_OPT_STATE
  else
    unset FEDVLA_GLOBAL_LR FEDVLA_P_START FEDVLA_P_END FEDVLA_NUM_STEPS \
          FEDVLA_WARMUP_FRAC FEDVLA_OPT_STATE
  fi
  (
    cd /root/HPT || exit 1
    "$HPT_PY" -m hpt.run \
      env=mujoco_metaworld \
      domains="$dom" \
      +datasetsize=metaworld_1 \
      "dataset_generator_func.env_names=[\"$env\"]" \
      "rollout_runner.env_names=[\"$env\"]" \
      dataset.episode_cnt=100 \
      train.total_iters="$iters" \
      train.total_epochs=1000 \
      train.freeze_trunk=False \
      warmup_lr.step="$warmup" \
      train.pretrained_dir=hf://liruiw/hpt-base \
      output_dir="$out" \
      > "$log" 2>&1
  )
  rc=$?
  echo "[---] $task 用时 $(( $(date +%s) - t0 )) 秒 rc=$rc"
  if [ ! -f "$out/model.pth" ]; then
    echo "[!!] $task 训练失败，日志尾部："; tail -n 30 "$log"; exit 1
  fi
  grep -m1 "Epoch size:" "$log" || true
  echo "[OK] $task 完成 -> $out/model.pth"
done

echo "EQUAL_BUDGET_DONE"
