#!/usr/bin/env bash
# FedVLA 门2 异构烟雾测试（2 轮 × 4 客户端）
#
# 目的：在烧 50 轮算力之前，验证联邦管线三件套都正确：
#   1) 续训真的生效（第 1 轮的客户端 trunk 与第 0 轮全局不同 —— run_fedavg.sh 内置自检）
#   2) 聚合真的产出可用的全局模型（trunk.pth / model.pth / extras.pth / config.yaml）
#   3) 异构键并集完整（全局 model.pth 能覆盖 task1/task2/task3 三个域的 stem/head/normalizer）
#
# 与正式实验的唯一差别：NUM_ROUNDS=2、输出目录独立，不污染 output/fedavg_reach。
set -u

export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

SCRIPT_DIR=/root/HPT/federated
OUT=/root/HPT/output/smoke_hetero

echo "===== 门2 异构烟雾测试 ====="
echo "输出目录: $OUT"
rm -rf "$OUT"

START_ROUND=0 NUM_ROUNDS=2 OUTPUT_ROOT="$OUT" \
  bash "$SCRIPT_DIR/run_fedavg.sh" 2>&1 | tee "$OUT/../smoke_hetero.log"

echo "===== 烟雾测试结束 rc=$? ====="
