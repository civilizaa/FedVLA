#!/usr/bin/env bash
# Hydra 配置解析验证：确认 +datasetsize=metaworld_1 不会把 domains 覆写回 task1。
# 用 --cfg job 只做配置合成，不加载数据、不占 GPU。
HPT_PY=/root/miniconda3/envs/hpt/bin/python
cd /root/HPT || exit 1

echo "===== 场景A：当前顺序（domains 在前，+datasetsize 在后）====="
$HPT_PY -m hpt.run \
  env=mujoco_metaworld \
  domains=mujoco_metaworld_task2 \
  +datasetsize=metaworld_1 \
  'dataset_generator_func.env_names=["push-v2"]' \
  --cfg job 2>&1 | grep -nE "^domains:|env_names|episode_cnt:|total_iters:|freeze_trunk|warmup" | head -20

echo "===== 场景B：+datasetsize 在前，domains 在后（建议顺序）====="
$HPT_PY -m hpt.run \
  env=mujoco_metaworld \
  +datasetsize=metaworld_1 \
  domains=mujoco_metaworld_task2 \
  'dataset_generator_func.env_names=["push-v2"]' \
  --cfg job 2>&1 | grep -nE "^domains:|env_names|episode_cnt:|total_iters:|freeze_trunk|warmup" | head -20

echo "===== 场景C：不加 datasetsize，只用 env 默认（4个 env）====="
$HPT_PY -m hpt.run \
  env=mujoco_metaworld \
  domains=mujoco_metaworld_task2 \
  --cfg job 2>&1 | grep -nE "^domains:|env_names|episode_cnt:" | head -20

echo "CFG_PROBE_DONE"
