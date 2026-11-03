# FedVLA 联邦实验复现指南

本文档说明 FedVLA 的联邦学习实验如何从零复现。所有参数都是**实测确定**的，不是估算。

---

## 0. 环境前提

| 项 | 要求 | 实测值 |
|---|---|---|
| OS | Linux（Ubuntu 22.04）| AutoDL 容器 |
| GPU | 单卡 ≥ 24GB | RTX 3090 24GB |
| Python | 3.10 | conda env `hpt` |
| 基础模型 | HPT-Base 12.6M | `hf://liruiw/hpt-base` |
| 仿真 | Meta-World (MuJoCo 2.1) | 需 `MUJOCO_GL=egl` |

HPT 依赖 `mujoco-py`（需编译），**原生 Windows 不可用**，必须 WSL2 或 Linux。

---

## 1. 数据准备（一次性，约 16 分钟/任务）

```bash
cd /root/HPT
bash federated/collect_data.sh
```

产出（必须落盘，否则每次训练重渲染 16 分钟）：

| domain | Meta-World env | episode 数 | epoch_size |
|---|---|---|---|
| `mujoco_metaworld_task1` | reach-v2 | 100 | 15 |
| `mujoco_metaworld_task2` | push-v2 | 99 | 19 |
| `mujoco_metaworld_task3` | button-press-topdown-v2 | 100 | 24 |

**校验缓存是否可用**（`data/action` 是 zarr 的 chunk **目录**，用 `-f` 判断会误报失败）：

```bash
test -d /root/HPT/data/zarr_mujoco_metaworld_task1_resnet_traj100/data/action && echo OK
```

**实测 epoch_size**（复刻 HPT 的 SequenceSampler 逻辑，含 task1==15 自检）：

```bash
python federated/probe_epoch_size.py
```

> ⚠️ task2 少 1 条是**随机 rollout 失败**（日志 `Number of samples: 100` → `Avg 99 traj length`），不是配置问题。

---

## 2. 打补丁（必做，否则联邦静默失效）

```bash
bash federated/repatch.sh
```

两个补丁点，缺一不可：

1. **`load_trunk` 支持本地路径** —— 原版只有 `hf://` 分支，本地 `trunk.pth` 是**静默空操作**。
   不打补丁 → 每轮客户端从 hf 原始权重重开 → 联邦空转且**不报错**。
2. **extras 载入方式** —— 必须用**顶层 ParameterDict 逐键赋值**，
   **绝对不能**用 `self.load_state_dict(extras, strict=False)`：
   `normalizer` 用的 `LinearNormalizer` 继承 `DictOfTensorMixin`，
   其 `_load_from_state_dict` 会**无条件重建** `self.params_dict`，
   extras 里没有 normalizer 的键 → normalizer 被清空 → 续训第一步就崩：
   `AttributeError: 'ParameterDict' object has no attribute 'state'`

---

## 3. 联邦训练

### 3.1 异构 FedAvg（正式实验）

```bash
bash federated/run_all_experiments.sh hetero
# 等价于：
# START_ROUND=0 NUM_ROUNDS=50 OUTPUT_ROOT=/root/HPT/output/fedavg_hetero \
#   CLIENT_DOMAINS_OVERRIDE="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1" \
#   CLIENT_ENVS_OVERRIDE="reach-v2 push-v2 button-press-topdown-v2 reach-v2" \
#   CLIENT_EPOCH_SIZE_OVERRIDE="15 19 24 15" \
#   bash federated/run_fedavg.sh
```

### 3.2 IID 健全性检查

```bash
bash federated/run_all_experiments.sh iid
```

4 个客户端全部训练 task1。**FedAvg 结果应 ≈ 集中式（0.98 ± 0.05）**。
差距大说明管线有 bug —— 这是跑正式实验前的必做检查。

---

## 4. 基线

```bash
# 上界参照（HPT 官方 Meta-World 配方：6000 步 = 400 epoch）
bash federated/run_centralized.sh all

# 等预算基线（步数 = 联邦客户端总步数：2250 / 2850 / 3600）
bash federated/run_centralized_equal_budget.sh
```

> ⚠️ **两组都要报**。只报 6000 步，审稿人会质疑"差距是不是纯粹因为联邦训得少"。
> 等预算基线的 `warmup_lr.step` 也要按本地总步数下调（563/713/900），不能用 1000。

---

## 5. 评估

```bash
# 联邦全局模型
MODEL_DIR=/root/HPT/output/fedavg_hetero/global_r49 \
DOMAIN=mujoco_metaworld_task1 ENVNAME=reach-v2 \
N_SEEDS=3 EPISODE_NUM=20 N_JOBS=1 \
bash federated/eval_model.sh

# 集中式（批量，输出 eval_centralized_<task>.log）
CENTRAL_ROOT=/root/HPT/output/centralized bash federated/eval_centralized.sh
```

统计口径：**3 seeds × 20 rollouts，报均值 ± 95% CI**（t 分布）。

```bash
# 提取成功率并算 CI
python federated/summarize_eval.py /root/HPT/output/eval_centralized_task1.log

# 生成论文对比表
bash federated/compare_results.sh
```

---

## 6. 通信分析（第 6 章）

```bash
python federated/comm_analysis.py --model /root/HPT/output/fedavg_hetero/global_r49/model.pth
python federated/comm_cost.py --model /root/HPT/output/fedavg_hetero/global_r49/model.pth
```

实测结果（50 轮 × 4 客户端）：

| 方案 | 通信量 | 压缩比 | 相对误差 |
|---|---|---|---|
| fp32 | 12.64 GB | 100% | 0 |
| int8 per-tensor | 3.16 GB | 25% | 0.0065 |
| **int8 per-group(64)** | **3.16 GB** | **25%** | **0.0046** |
| int4 per-group(64) | 1.58 GB | 12.5% | 0.0833 |
| int2 per-group(64) | 0.79 GB | 6.2% | 0.5678 |
| 1-bit 符号化 | 0.40 GB | 3.1% | 0.5450 |

共享 trunk **12,644,352** 参数；个性化 stem/head **3,892,838** 参数
→ **联邦只传 76.5% 的参数**。

---

## 7. 出图

```bash
python federated/extract_convergence.py --root /root/HPT/output/fedavg_hetero
python federated/make_figures.py --all
```

产出到 `/root/HPT/output/figures/`：
`fig4_convergence.png` / `fig5_param_share.png` / `fig6_comparison.png` / `fig7_comm_cost.png`

---

## 8. 关键实现细节（踩过的坑）

### 8.1 `train.total_iters` 的单位是 **batch step 数**，不是 epoch 数

源码 `hpt/run.py`：

```python
train_steps = (epoch + 1) * len(train_loader)
if train_steps > cfg.train.total_iters:
    break
```

跑满 N 个 epoch → `total_iters = N × epoch_size - 1`（减 1 是因为判断是严格大于）。

### 8.2 `warmup_lr.step` 必须按本地总步数下调

`WarmupLR` 从 `init_lr=0` **线性升到**目标 lr，升满需要 `warmup_lr.step` 步。
默认值 1000 是为集中式 6000 步设计的；联邦本地只有几十步时，
整个本地训练期间 lr 都近似 0 → 本地模型几乎不变 → 联邦"看起来没效果"，
**但其实是配置错误，不是方法失效**。联邦取本地总步数的 1/4。

### 8.3 `rollout_runner.env_names` 必须显式覆盖

`+datasetsize=metaworld_1` 里它是 `["reach-v2"]`，不覆盖就会**在错误的 env 上评估**。
用 `python -m hpt.run --cfg job` 可验证合成后的实际取值。

### 8.4 bash 数组不能用 `${VAR:-"a" "b"}` 赋值

那会把三个值存成**单个元素**（`${A[1]}` 为空），且 `set -u` 下未定义变量直接报错。
必须用 `if/else` + `read -r -a`。

### 8.5 异构聚合必须覆盖所有非 trunk 键

HPT 完整 state_dict 的顶层前缀分布：
`trunk(224) / stems(18) / normalizer(12) / heads(8) / modalities_tokens(2)`。

`normalizer.<domain>`、`modalities_tokens.<modality>` 也是按域/模态分开存的。
只并集 stems/heads 会导致全局模型缺少其它域的 normalizer → eval 时直接崩。

### 8.6 聚合必须是真平均，不能是拷贝

验证脚本 `verify_global.py` 会检查：
全局 trunk 与 N 个客户端均值的偏差应 = 0，与单个客户端的偏差应 > 0。

---

## 9. 一键跑完

```bash
bash federated/run_queue.sh
```

自动依次执行：等预算基线 → 集中式评估 → 等预算评估 → IID 健全性 → IID 评估 → 对比表。
每步之间等 GPU 空闲。
