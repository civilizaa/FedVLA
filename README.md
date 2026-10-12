# FedVLA 实验仓库

通信硕士论文《面向通信约束的联邦视觉-语言-动作模型》的实验代码与云环境配置。
主线：**联邦学习（FL）** 训练 VLA 策略；Split Learning 作为第 5 章增量扩展（SplitFed）。

> 本仓库是**唯一可靠备份**。云实例随时可能被重置，代码必须每天 push 到这里。

## 目录结构

```
fedvla-repo/
├── setup_cloud.sh        # 云服务器一键环境安装（幂等，可重复跑）
├── fedvla_scaffold/      # numpy 版 FedVLA 核心算法（已跑通，验证数学用）
│   ├── fl_core.py        # DGMoE / EDA / FedAvg 核心
│   └── smoke_test.py     # 三组冒烟测试
├── federated/            # 【待建】联邦训练主循环、聚合策略、客户端
├── comms/                # 【待建】通信损伤：丢包注入、量化压缩、流量预算
├── privacy/              # 【待建】差分隐私（DP-SGD）、特征反演评估
├── experiments/          # 【待建】实验矩阵配置、评测脚本、结果汇总
├── docs/
│   └── planning/         # 选题与执行规划文档（路线图、可行性、环境卡等）
├── env_setup.md          # 踩坑日志：每次环境问题都记在这里
└── .gitignore
```

## 快速开始（云服务器）

```bash
# 1. 克隆
git clone https://github.com/<你的用户名>/fedvla-repo.git
cd fedvla-repo

# 2. 一键装环境（建议在「无卡模式」下跑，约 30–40 分钟）
bash setup_cloud.sh

# 3. 验证内置 Meta-World
python -c "from env.mujoco.metaworld.rollout_runner import ALL_TASK_CONFIG; print(len(ALL_TASK_CONFIG))"
# 输出 50 即通

# 4. 1task 训练冒烟（HPT 仓库需另行克隆到同级目录，见 env_setup.md）
cd ../HPT
python -m hpt.run env=mujoco_metaworld domains=mujoco_metaworld_task1 \
  +datasetsize=metaworld_1 dataset.episode_cnt=100 \
  train.pretrained_dir=hf://liruiw/hpt-base +mode=debug
```

## 持久化三铁律

1. **代码**：每天结束前 `git add -A && git commit && git push`。
2. **checkpoint**：训练产物放**数据盘**（不是系统盘），云重置不丢。
3. **results**：每周五把 `experiments/` 下的结果打包下载到本地 Windows。

## 关键约束（来自规划文档）

- 训练全在云/Linux，写作留本地 Windows。
- **不要** `pip install metaworld`——HPT 内置了一份，PyPI 版会打架。
- HPT 用 `--no-deps` 安装，避免拉 tensorflow/rlds/zarr/dm-reverb/ray。
- 显存/算力充裕时：seeds 提到 5，每次只改一项参数并记进实验日志。

详见 `docs/planning/` 与各模块 README。
