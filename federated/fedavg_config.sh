#!/usr/bin/env bash
# FedVLA 联邦训练（FedAvg 基线）配置 —— 所有旋钮集中在这里。
# 远程执行前只需改这个文件；run_fedavg.sh 会 source 它。

# ===== 路径（全部用绝对路径，避免新 shell 里 conda/python 失效）=====
# 注意：全部用 ${VAR:-default} 形式，允许用环境变量覆盖（烟雾测试/正式实验共用同一份脚本）
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/HPT/output/fedavg_reach}   # 全局/客户端输出根目录
HPT_PY=/root/miniconda3/envs/hpt/bin/python    # hpt 环境的绝对路径 python
FED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # 本脚本所在目录（federated/）

# HPT 的 datasetsize 配置组。metaworld_1 = 每任务 1 个 env 采样器。
# ⚠️ 它同时决定 dataset_generator_func.env_names 与 rollout_runner.env_names，
#    所以每个客户端必须用命令行覆盖成自己任务的 env（见 run_fedavg.sh）。
# ⚠️ metaworld_1.yaml 末尾带 `domains: mujoco_metaworld_task1`，
#    但 Hydra 的命令行覆盖优先级高于配置文件 -> domains 仍正确（已用 --cfg job 实测）。
DATASETSIZE=${DATASETSIZE:-metaworld_1}

# 训练期的验证 rollout 次数（metaworld_1.yaml 默认 30）。
# 这里调小是为了让本地训练的验证开销不占满算力；正式评估走 eval_model.sh（3 seeds × 20 rollouts）。
TRAIN_EVAL_ROLLOUTS=${TRAIN_EVAL_ROLLOUTS:-5}

# ===== 联邦超参 =====
NUM_ROUNDS=${NUM_ROUNDS:-50}
LOCAL_EPOCHS=${LOCAL_EPOCHS:-3}                      # 每客户端本地 epoch 数（目标）

# ⚠️ total_iters 的单位是【总 batch step 数】，不是 epoch 数！
# 源码依据 hpt/run.py：
#     train_steps = (epoch + 1) * len(train_loader)      # 每个 epoch 走 len(train_loader) 步
#     if train_steps > cfg.train.total_iters: break       # 超过才停
# 所以"跑满 LOCAL_EPOCHS 个 epoch"需要：
#     total_iters = LOCAL_EPOCHS × epoch_size - 1
# （减 1 是因为判断是严格大于；写成 LOCAL_EPOCHS×epoch_size 会多跑一个 epoch）
#
# epoch_size = len(train_loader) = ceil(len(dataset)/256)（batch_size=256, drop_last=False）
# 不同 Meta-World 任务的 episode 长度不同 -> epoch_size 不同 -> 必须逐任务测量。
# 由 federated/probe_epoch_size.py 实测（复刻 HPT 源码的 SequenceSampler 逻辑，
# 并用集中式门1 的 task1=15 做自检，不一致会直接报错退出）。
# 实际数组赋值在下方"客户端任务分配"一节（避免 bash 数组默认值陷阱）。

EPISODE_CNT=100                                 # 每任务示范数（论文协议：100 条/任务）

# 初始预训练（第 0 轮从 HPT 官方权重开始；第 1 轮起由脚本自动改为上一轮 global 目录）
INIT_PRETRAINED=hf://liruiw/hpt-base

# HPT 训练额外参数。trunk 必须可训（联邦要聚合它）-> freeze_trunk=False。
#
# ⚠️ warmup_lr.step 必须按【本地总步数】下调！
# 源码 hpt/utils/warmup_lr_wrapper.py：WarmupLR 从 init_lr=0【线性升到目标 lr】，
# 升满需要 warmup_lr.step 步。默认值 1000 是为集中式 6000 步设计的；
# 联邦本地只有几十步，若还用 1000，整个本地训练期间 lr 都近似 0
# -> 本地模型几乎不变 -> 联邦"看起来没效果"，但其实是配置错误，不是方法失效。
# 这里取本地总步数的 1/4（向上取整），保证有足够步数在有效学习率区间。
WARMUP_RATIO_NUM=4

# HPT 没有 freeze_stem 方法；stem 是否可训取决于 cfg.stem，trunk 必须可训。

# ===== 客户端任务分配（异构核心）=====
# 每个字符串是一个客户端的 HPT `domains`（HPT 只用列表里第一个域训练）。
# 异构基线：4 客户端分 3 任务（最后一个与第一个同任务 -> 测"任务重叠客户端"）。
#
# ⚠️ domain 名 ↔ Meta-World 任务的对应关系（已核对 HPT 配置
#    experiments/configs/env/mujoco_metaworld.yaml 的 env_names 顺序）：
#      task1 = reach-v2                （已验证）
#      task2 = push-v2
#      task3 = button-press-topdown-v2
#      task4 = door-open-v2
# CLIENT_ENVS 必须与 CLIENT_DOMAINS 一一对应：它会被写进
# `dataset_generator_func.env_names`，万一缓存缺失要重新采集时，
# HPT 才会生成【正确任务】的数据（否则 task2 客户端会生成 reach-v2 的数据！）。
# ⚠️ bash 陷阱：数组不能用 ${VAR:-"a" "b"} 赋值 —— 那会把 "a b c" 存成【单个元素】。
#    必须用 if/else 显式赋值，才能既保留默认数组、又能被环境变量整体覆盖。
if [ -n "${CLIENT_DOMAINS_OVERRIDE:-}" ]; then
  read -r -a CLIENT_DOMAINS <<< "$CLIENT_DOMAINS_OVERRIDE"
else
  CLIENT_DOMAINS=( "mujoco_metaworld_task1" "mujoco_metaworld_task2" "mujoco_metaworld_task3" "mujoco_metaworld_task1" )
fi

if [ -n "${CLIENT_ENVS_OVERRIDE:-}" ]; then
  read -r -a CLIENT_ENVS <<< "$CLIENT_ENVS_OVERRIDE"
else
  CLIENT_ENVS=( "reach-v2" "push-v2" "button-press-topdown-v2" "reach-v2" )
fi

if [ -n "${CLIENT_EPOCH_SIZE_OVERRIDE:-}" ]; then
  read -r -a CLIENT_EPOCH_SIZE <<< "$CLIENT_EPOCH_SIZE_OVERRIDE"
else
  CLIENT_EPOCH_SIZE=( 15 19 24 15 )
fi

# —— IID 健全性检查（强烈建议先跑这个验证整条管线）——
# 把上面三行换成全部 task1：FedAvg 应≈集中式(~0.98)；若差距大说明管线有 bug。
# 不用改文件，直接用环境变量：
#   CLIENT_DOMAINS_OVERRIDE="mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1 mujoco_metaworld_task1" \
#   CLIENT_ENVS_OVERRIDE="reach-v2 reach-v2 reach-v2 reach-v2" \
#   CLIENT_EPOCH_SIZE_OVERRIDE="15 15 15 15" bash run_fedavg.sh

# FedAvg 聚合权重：equal=等权（默认）；data=按本地样本数(=epoch_size)加权（教科书 FedAvg）
FEDAVG_WEIGHTS=${FEDAVG_WEIGHTS:-equal}

NUM_CLIENTS=${#CLIENT_DOMAINS[@]}

# ===== 配置自检（错了就别烧算力）=====
if [ "${#CLIENT_ENVS[@]}" -ne "$NUM_CLIENTS" ]; then
  echo "[!!] CLIENT_ENVS 数量(${#CLIENT_ENVS[@]}) != CLIENT_DOMAINS 数量($NUM_CLIENTS)"
  exit 1
fi
if [ "${#CLIENT_EPOCH_SIZE[@]}" -ne "$NUM_CLIENTS" ]; then
  echo "[!!] CLIENT_EPOCH_SIZE 数量(${#CLIENT_EPOCH_SIZE[@]}) != CLIENT_DOMAINS 数量($NUM_CLIENTS)"
  exit 1
fi

# 客户端 i 的 domain 必须与 env 一一对应（错配会让客户端去训练别的任务的数据）。
# 映射表来自 experiments/configs/env/mujoco_metaworld.yaml 的 env_names 顺序。
domain_env_of () {
  case "$1" in
    mujoco_metaworld_task1) echo "reach-v2" ;;
    mujoco_metaworld_task2) echo "push-v2" ;;
    mujoco_metaworld_task3) echo "button-press-topdown-v2" ;;
    mujoco_metaworld_task4) echo "door-open-v2" ;;
    *) echo "" ;;
  esac
}
for ((c=0; c<NUM_CLIENTS; c++)); do
  expected_env=$(domain_env_of "${CLIENT_DOMAINS[$c]}")
  if [ -z "$expected_env" ]; then
    echo "[!!] client $c 的 domain '${CLIENT_DOMAINS[$c]}' 不在已知映射表里"
    exit 1
  fi
  if [ "$expected_env" != "${CLIENT_ENVS[$c]}" ]; then
    echo "[!!] client $c 错配：domain=${CLIENT_DOMAINS[$c]} 应配 env=$expected_env，" \
         "但配置写的是 ${CLIENT_ENVS[$c]}"
    exit 1
  fi
done
echo "[OK] 配置自检通过：$NUM_CLIENTS 个客户端，domain↔env 映射一致"
