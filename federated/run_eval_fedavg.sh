#!/usr/bin/env bash
# 评估联邦全局模型（与集中式基线同一口径：3 seeds × 20 rollouts）。
#
# 用法：
#   GLOBAL_DIR=/root/HPT/output/fedavg_reach/global_r49 bash run_eval_fedavg.sh
#   GLOBAL_DIR=... N_JOBS=1 bash run_eval_fedavg.sh   # 8GB 卡先单进程
#
# 注意：--config-path 必须是【绝对路径】且指向含 config.yaml 的全局目录
#       （HPT 的 --config-path 相对包目录解析，不是 cwd；之前踩过坑）。
#       全局目录由 aggregate.py 写出 model.pth + config.yaml + config.json，满足要求。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPT_PY=/root/miniconda3/envs/hpt/bin/python

GLOBAL_DIR="${GLOBAL_DIR:?请设置 GLOBAL_DIR 为全局模型目录}"
N_JOBS="${N_JOBS:-3}"
EPISODE_NUM="${EPISODE_NUM:-20}"

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

SWEEP_DIR="${GLOBAL_DIR}_eval"
LOG="${GLOBAL_DIR}_eval.log"

nohup env OMP_NUM_THREADS=8 MUJOCO_GL=egl WANDB_MODE=offline HF_ENDPOINT=https://hf-mirror.com \
  "$HPT_PY" -m hpt.run_eval --multirun \
  --config-name=config \
  --config-path="$GLOBAL_DIR" \
  train.pretrained_dir="$GLOBAL_DIR" \
  seed="range(3)" \
  rollout_runner.episode_num="$EPISODE_NUM" \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs="$N_JOBS" \
  hydra.sweep.dir="$SWEEP_DIR" \
  > "$LOG" 2>&1 &
echo "eval launched pid $!  ->  tail -f $LOG"
echo "完成后看结果: grep -iE 'success|rate|seed|mean|avg' $LOG | tail -n 30"
