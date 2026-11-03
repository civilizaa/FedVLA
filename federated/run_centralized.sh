#!/usr/bin/env bash
# FedVLA：集中式基线（每任务一个模型，无联邦）—— 联邦结果的【上界参照】。
#
# 用法：
#   bash run_centralized.sh task1            # 只跑 task1
#   bash run_centralized.sh all              # 三个任务串行
#   ITERS=2250 bash run_centralized.sh task1 # 等预算基线（与联邦客户端总步数一致）
#
# ⚠️ 预算口径（论文要写清楚）：
#   集中式 = 6000 步 = 400 epoch（epoch_size=15 时），这是 HPT 官方 Meta-World 微调配方。
#   联邦   = 50 轮 × 3 本地 epoch × epoch_size(15) = 2250 步 = 150 epoch。
#   两者不是等预算 —— 集中式是"上界参照"（同样的 100 条示范，
#   但把数据集中起来训练），联邦是"通信受限下的分布式"。
#   若要做严格等预算对比，用 ITERS=2250。
set -u

HPT_PY=/root/miniconda3/envs/hpt/bin/python
OUT_ROOT=${OUT_ROOT:-/root/HPT/output/centralized}
ITERS=${ITERS:-6000}
WARMUP=${WARMUP:-1000}
EPISODE_CNT=${EPISODE_CNT:-100}
TARGET=${1:-all}

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# domain -> Meta-World env（已核对 experiments/configs/env/mujoco_metaworld.yaml）
domain_of () {
  case "$1" in
    task1) echo "mujoco_metaworld_task1" ;;
    task2) echo "mujoco_metaworld_task2" ;;
    task3) echo "mujoco_metaworld_task3" ;;
    *) echo "" ;;
  esac
}
env_of () {
  case "$1" in
    task1) echo "reach-v2" ;;
    task2) echo "push-v2" ;;
    task3) echo "button-press-topdown-v2" ;;
    *) echo "" ;;
  esac
}

run_one () {
  local t="$1"
  local dom env out log
  dom=$(domain_of "$t"); env=$(env_of "$t")
  if [ -z "$dom" ]; then echo "[!!] 未知任务 $t"; return 1; fi
  out="$OUT_ROOT/$t"
  log="$OUT_ROOT/train_$t.log"
  mkdir -p "$OUT_ROOT"

  # 续跑保护：已有 model.pth 就跳过
  if [ -f "$out/model.pth" ]; then
    echo "[SKIP] $t 已训好：$out/model.pth"
    return 0
  fi

  echo "===== 集中式 $t  domain=$dom env=$env  iters=$ITERS -> $out ====="
  local t0=$(date +%s)
  (
    cd /root/HPT || exit 1
    "$HPT_PY" -m hpt.run \
      env=mujoco_metaworld \
      domains="$dom" \
      +datasetsize=metaworld_1 \
      "dataset_generator_func.env_names=[\"$env\"]" \
      dataset.episode_cnt="$EPISODE_CNT" \
      train.total_iters="$ITERS" \
      train.total_epochs=1000 \
      train.freeze_trunk=False \
      warmup_lr.step="$WARMUP" \
      train.pretrained_dir=hf://liruiw/hpt-base \
      output_dir="$out" \
      > "$log" 2>&1
  )
  local rc=$?
  echo "[---] $t 用时 $(( $(date +%s) - t0 )) 秒 rc=$rc"
  if [ ! -f "$out/model.pth" ]; then
    echo "[!!] $t 训练失败，日志尾部："; tail -n 30 "$log"; return 1
  fi
  grep -m1 "Epoch size:" "$log" || true
  echo "[OK] $t 完成 -> $out/model.pth"
}

if [ "$TARGET" = "all" ]; then
  for t in task1 task2 task3; do run_one "$t" || exit 1; done
else
  run_one "$TARGET" || exit 1
fi
echo "CENTRALIZED_DONE"
