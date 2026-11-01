#!/usr/bin/env bash
# FedVLA：采集异构客户端所需的 Meta-World 任务数据（现场渲染 + resnet 特征预计算）
# 用法：bash collect_data.sh
#
# 关键事实（已核实 HPT 源码）：
#   - LocalTrajDataset 数据来源 = 本地缓存 data/zarr_<domain>_<encoder>_traj<cnt>
#     不存在则调用 cfg.dataset_generator_func 现场渲染；不从 HF 下载。
#   - generate_dataset_rollouts 只认 env_names，不认 domains。
#     domains 只是缓存目录标签，env_names 决定真实任务。
#   - 所以：domains=mujoco_metaworld_taskN + env_names=[<真实env>] 即可任意配对。
set -u
cd /root/HPT || exit 1
export MUJOCO_GL=egl WANDB_MODE=offline HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/hpt/bin/python

# ---- 防重入锁 ----
# relay 队列重派 / 手动重复执行时，若已有采集进程在跑，直接退出，
# 否则两个进程会同时写同一个 zarr 目录，把数据写坏。
exec 9>/root/.collect_data.lock
if ! flock -n 9; then
  echo "[SKIP] 已有采集进程在运行（锁被占用），本次跳过"
  exit 0
fi

collect () {
  local domain="$1" env="$2"
  local cache="/root/HPT/data/zarr_${domain}_resnet_traj100"
  if [ -f "$cache/data/action" ]; then
    echo "[SKIP] $domain 缓存已存在：$cache"
    return 0
  fi
  echo "=========== 采集 $domain <- $env ==========="
  local t0=$(date +%s)
  "$PY" -m hpt.run \
    env=mujoco_metaworld \
    domains="$domain" \
    +datasetsize=metaworld_1 \
    "dataset_generator_func.env_names=[\"$env\"]" \
    dataset.episode_cnt=100 \
    train.total_iters=1 train.total_epochs=1 \
    train.epoch_iters=1 train.validation_iters=1 \
    train.pretrained_dir=hf://liruiw/hpt-base \
    output_dir="/root/HPT/output/_collect_${domain}" \
    2>&1 | grep -vE "^(wandb:| *$)" | tail -25
  echo "[---] 用时 $(( $(date +%s) - t0 )) 秒"
  if [ -f "$cache/data/action" ]; then
    echo "[OK] $domain 数据就绪"
    du -sh "$cache"
  else
    echo "[!!] $domain 采集失败"
  fi
}

collect mujoco_metaworld_task2 push-v2
collect mujoco_metaworld_task3 button-press-topdown-v2
echo "ALL_COLLECT_DONE"
