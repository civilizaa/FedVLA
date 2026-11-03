#!/usr/bin/env bash
# 打印 Hydra 合成后的关键字段实际取值（列表在下一行，需 -A1）
HPT_PY=/root/miniconda3/envs/hpt/bin/python
cd /root/HPT || exit 1

dump () {
  echo "----- $1 -----"
  shift
  $HPT_PY -m hpt.run "$@" --cfg job 2>&1 | grep -nE -A1 "^domains:|^  env_names:|^dataset_generator_func:|^rollout_runner:" | head -30
}

dump "A: domains前 +datasetsize后（当前写法）" \
  env=mujoco_metaworld \
  domains=mujoco_metaworld_task2 \
  +datasetsize=metaworld_1 \
  'dataset_generator_func.env_names=["push-v2"]' \
  'rollout_runner.env_names=["push-v2"]'

dump "B: 只覆盖 dataset_generator_func，不覆盖 rollout_runner" \
  env=mujoco_metaworld \
  domains=mujoco_metaworld_task2 \
  +datasetsize=metaworld_1 \
  'dataset_generator_func.env_names=["push-v2"]'

echo "ENVNAME_PROBE_DONE"
