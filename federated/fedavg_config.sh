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

# ⚠️ total_iters 的单位是【总 batch step 数】，不是 epoch 数！
# 源码依据 hpt/run.py：
#     train_steps = (epoch + 1) * len(train_loader)      # 每个 epoch 走 len(train_loader) 步
#     if train_steps > cfg.train.total_iters: break       # 超过才停
# 所以"跑满 LOCAL_EPOCHS 个 epoch"需要：
#     total_iters = LOCAL_EPOCHS × epoch_size - 1
# （减 1 是因为判断是严格大于；写成 LOCAL_EPOCHS×epoch_size 会多跑一个 epoch）
#
# epoch_size = len(train_loader) = ceil(训练集样本数 / batch_size)，随任务/数据量变化。
# 集中式门1 实测：epoch_size=15（batch 256、100 episode、task1）。
# 三个任务数据量相近，异构基线统一按 15 起步，**跑第一轮后必须核对日志里的
# "Epoch size:" 行**，若不同任务的 epoch_size 不一致要按任务分别设置。
LOCAL_EPOCH_SIZE=15                            # TODO: 用第一轮日志的 "Epoch size:" 核对
LOCAL_TOTAL_ITERS=$((LOCAL_EPOCHS * LOCAL_EPOCH_SIZE - 1))   # = 44

EPISODE_CNT=100                                 # 每任务示范数（与集中式基线一致）
DATASETSIZE=metaworld_1                         # 单任务数据集；多任务客户端见 README

# 初始预训练（第 0 轮从 HPT 官方权重开始；第 1 轮起由脚本自动改为上一轮 global 目录）
INIT_PRETRAINED=hf://liruiw/hpt-base

# HPT 训练额外参数。若远程确认有冻结 stem 的 flag（如 train.freeze_stem=True），可加在这里；
# 注意：HPT 没有 freeze_stem 方法，stem 是否可训取决于 cfg.stem；trunk 必须可训（freeze_trunk=False）。
#
# ⚠️ warmup_lr.step 必须下调！
# 源码 hpt/utils/warmup_lr_wrapper.py：WarmupLR 是从 init_lr=0 【线性升到目标 lr】，
# 升满需要 warmup_lr.step 步。默认值 1000 是为集中式 6000 步设计的；
# 联邦本地只有 ~44 步，若还用 1000，则整个本地训练期间 lr 都近似 0
# → 本地模型几乎不变 → 联邦"看起来没效果"，但其实是配置错误，不是方法失效。
# 这里取本地总步数的 1/4（向上取整），保证有足够步数在有效学习率区间。
LOCAL_WARMUP=$(( (LOCAL_TOTAL_ITERS + 3) / 4 ))
EXTRA_HPT_ARGS="train.freeze_trunk=False warmup_lr.step=${LOCAL_WARMUP}"

# ===== 客户端任务分配（异构核心）=====
# 每个字符串是一个客户端的 HPT `domains`（HPT 只用列表里第一个域训练）。
# 异构基线：4 客户端分 3 任务（最后一个是重叠/重复客户端）。
# 任务 domain 名请在远程确认（reach=mujoco_metaworld_task1 已验证；其余用 partition.py 或 HPT 配置查）。
CLIENT_DOMAINS=( "mujoco_metaworld_task1" "mujoco_metaworld_task2" "mujoco_metaworld_task3" "mujoco_metaworld_task1" )

# —— IID 健全性检查（强烈建议先跑这个验证整条管线）——
# 把上面一行换成全部 task1：FedAvg 应≈集中式(~0.98)；若差距大说明管线有 bug。
# CLIENT_DOMAINS=( "mujoco_metaworld_task1" "mujoco_metaworld_task1" "mujoco_metaworld_task1" "mujoco_metaworld_task1" )

NUM_CLIENTS=${#CLIENT_DOMAINS[@]}
