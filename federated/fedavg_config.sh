#!/usr/bin/env bash
# FedVLA 联邦训练（FedAvg 基线）配置 —— 所有旋钮集中在这里。
# 远程执行前，按注释把 ?? 占位填好（尤其是 LOCAL_TOTAL_ITERS 和任务 domain 名）。

# ===== 路径（全部用绝对路径，避免新 shell 里 conda/python 失效）=====
OUTPUT_ROOT=/root/HPT/output/fedavg_reach      # 全局/客户端输出根目录
HPT_PY=/root/miniconda3/envs/hpt/bin/python    # hpt 环境的绝对路径 python
FED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 本脚本所在目录（federated/）

# ===== 联邦超参 =====
NUM_ROUNDS=50
LOCAL_EPOCHS=3                                 # 每客户端本地 epoch 数（目标）

# 本地步数 = LOCAL_EPOCHS × 单 epoch 步数。
# 先用【一个】客户端跑一次（见 README 校准步骤），看日志里 "Epoch size: X"，
# 把下面 LOCAL_TOTAL_ITERS 改成 3*X。若不想分步校准，可直接给一个经验值。
LOCAL_TOTAL_ITERS=3000                          # TODO: 校准后替换

EPISODE_CNT=100                                 # 每任务示范数（与集中式基线一致）
DATASETSIZE=metaworld_1                         # 单任务数据集；多任务客户端见 README

# 初始预训练（第 0 轮从 HPT 官方权重开始；第 1 轮起由脚本自动改为上一轮 global 目录）
INIT_PRETRAINED=hf://liruiw/hpt-base

# HPT 训练额外参数。若远程确认有冻结 stem 的 flag（如 train.freeze_stem=True），可加在这里；
# 注意：HPT 没有 freeze_stem 方法，stem 是否可训取决于 cfg.stem；trunk 必须可训（freeze_trunk=False）。
EXTRA_HPT_ARGS="train.freeze_trunk=False"

# ===== 客户端任务分配（异构核心）=====
# 每个字符串是一个客户端的 HPT `domains`（HPT 只用列表里第一个域训练）。
# 异构基线：4 客户端分 3 任务（最后一个是重叠/重复客户端）。
# 任务 domain 名请在远程确认（reach=mujoco_metaworld_task1 已验证；其余用 partition.py 或 HPT 配置查）。
CLIENT_DOMAINS=( "mujoco_metaworld_task1" "mujoco_metaworld_task2" "mujoco_metaworld_task3" "mujoco_metaworld_task1" )

# —— IID 健全性检查（强烈建议先跑这个验证整条管线）——
# 把上面一行换成全部 task1：FedAvg 应≈集中式(~0.98)；若差距大说明管线有 bug。
# CLIENT_DOMAINS=( "mujoco_metaworld_task1" "mujoco_metaworld_task1" "mujoco_metaworld_task1" "mujoco_metaworld_task1" )

NUM_CLIENTS=${#CLIENT_DOMAINS[@]}
