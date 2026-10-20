# federated/ — 联邦训练主模块（FedAvg 基线 + 后续 EDA/按簇）

第 2 阶段落地（联邦框架 + FedAvg 基线，门 2：复现"异构有害"）。
后续 H1/H2（EDA、按簇聚合）在同一套 trunk 聚合接口上扩展。

---

## 1. 设计总览

**核心思路：checkpoint averaging（权重平均），复用 HPT 自带训练，不重写训练循环。**

- 每轮：4 个客户端**串行**在 GPU 上本地训 3 epoch（冻结逻辑由 HPT 控制，trunk 必须可训）。
- 聚合：把各客户端 `model.pth` 读出，**只平均 trunk 参数**（键前缀 `trunk.`），写回全局目录
  的 `trunk.pth` + `model.pth` + 复制 `config.yaml`/`config.json`。
- 下一轮：客户端从全局目录**续训**（HPT `init_policy` 靠 `trunk.pth` 走本地续训分支）。

为什么只聚合 trunk：HPT 的 stem/head 是**按域的 ModuleDict**（`stems.<domain>_*`、`heads.<domain>`），
不同客户端的域不同，无法跨客户端平均，且 FedVLA 本就是"trunk 共享聚合、stem/head 个性化"。
所以全局模型 = 平均 trunk + 各客户端 stem/head 并集。

---

## 2. 文件清单

| 文件 | 作用 |
|---|---|
| `partition.py` | 任务级非 iid 划分（纯逻辑，可本地自测）。产出 `CLIENT_DOMAINS` 数组。 |
| `fedavg_config.sh` | **所有旋钮集中地**：轮数、本地步数、任务分配、路径。 |
| `run_fedavg.sh` | 编排：轮×客户端的训练 + 每轮聚合；支持 `START_ROUND` 断点续训。 |
| `aggregate.py` | 服务器端 FedAvg：平均 trunk，写全局 `trunk.pth`/`model.pth`/`config*`。 |
| `run_eval_fedavg.sh` | 评估全局模型（与集中式基线同一口径：3 seeds × 20 rollouts）。 |
| `verify_continuation.py` | **运行前必做**：验证 HPT 能从本地 `trunk.pth` 续训。 |

---

## 3. 从 HPT 源码核实的关键事实（写代码时锁死）

1. **续训靠 `trunk.pth`**：`init_policy` 判断 `os.path.exists(pretrained_dir/trunk.pth)`，有则走本地续训，并 `load_trunk(trunk.pth)` + 读 `config.yaml` 实例化同架构。
2. **`domains` 是逗号分隔字符串**（`cfg.domains.split(",")`）；但**单次训练只用列表第一个域**构建数据集与 stem/head。→ 每个客户端挂**一个**任务，异构靠"不同客户端挂不同任务"。
3. **数据是离线 Zarr 缓存**（`data/zarr_<name>`），首次落地后**复用，不每轮重采** → 50 轮×4 客户端时长可控。
4. **`train.freeze_trunk`**：默认冻结 trunk（20 任务脚本显式设 `False` 解冻）。联邦必须训 trunk → 配置里设 `train.freeze_trunk=False`。
5. **没有 `freeze_stem` 方法**；stem 是否可训取决于 `cfg.stem`。stem/head 不跨客户端聚合，对 FedAvg 数学无影响（只做个性化），故不强求冻结 stem。
6. **`model.pth` = 完整 state_dict**（eval 用它）；`trunk.pth` = trunk-only（续训用它）。聚合脚本两者都写。

---

## 4. 运行前必做：验证本地续训（30 秒）

按顺序做 3 个小验证，**任一不过都不启动 50 轮正式训练**。

### 0. 先确认 trunk 键名格式（10 秒）

`aggregate.py` 假设：完整 `model.pth` 里 trunk 参数键以 `trunk.` 开头，聚合时去掉这一层得到 `trunk.pth`
（供 `policy.load_trunk` 用）。如果 HPT 版本把 trunk 包成了嵌套子模块（键形如 `trunk.trunk.xxx`），
当前 `extract_trunk_only` 去掉首层 `trunk.` 仍正确；但若键名不是这个规律就会写错。先打印真实键名确认：

```bash
cd ~/HPT
/root/miniconda3/envs/hpt/bin/python - <<'PY'
import torch
sd = torch.load("output/reach_baseline/model.pth", map_location="cpu", weights_only=False)
tk = [k for k in sd if k.startswith("trunk")]
print("trunk 键示例:", tk[:6])
print("trunk 键数量:", len(tk))
PY
```
- 期望看到 `trunk.xxx` 或 `trunk.trunk.xxx` 这类以 `trunk.` 起头的键，且数量在几百到上千。
- 如果 trunk 键不以 `trunk.` 开头（比如直接 `xxx`），则 `aggregate.py` 的 `_is_trunk_key` 会漏选，
  需改 `average_trunk` 的判定（届时按打印结果调整）。

### 1. 验证 load_trunk 本地加载（verify_continuation.py）

`load_trunk` 我扒到的源码只显示了 `hf://` 分支，本地路径是否真能加载未 100% 确定。
**先在远程 hpt 环境跑一次**：

```bash
cd ~/HPT
/root/miniconda3/envs/hpt/bin/python federated/verify_continuation.py
```

- 输出 `LOADED_TRUNK_MATCHES_SOURCE = True` → 联邦管线可直接用。
- 输出 `False` → `load_trunk` 本地是空操作。兜底方案见第 8 节（自定义循环直接 `load_state_dict` 全局 `model.pth`）。

### 2. 异构续训烟雾测试（1 轮，10 分钟级）

**最关键、最容易被忽略**：IID 健全性（全 task1）只验证了"同域续训"，但正式异构里
客户端在 task2/task3 上训练、却要加载一个 **task1 全局 config.yaml 的目录**。需确认 HPT 用命令行
`domains=task2` 覆盖架构、能正确新建 task2 的 stem/head 并完成续训，而不是沿用 config.yaml 里的 task1 架构。

做法：把 `fedavg_config.sh` 的 `NUM_ROUNDS` 临时改成 `1`，`CLIENT_DOMAINS` 用异构分配
`(task1 task2 task3 task1)`，跑一轮；看 4 个客户端是否都训出 `model.pth`、聚合是否产出 `trunk.pth`。
全过再改回 `NUM_ROUNDS=50` 跑正式。

```bash
cd ~/HPT
# 编辑 fedavg_config.sh：NUM_ROUNDS=1，CLIENT_DOMAINS=( task1 task2 task3 task1 )
bash federated/run_fedavg.sh
ls output/fedavg_reach/global_r0/   # 应有 trunk.pth / model.pth / config.yaml
```

---

## 5. 校准本地步数（一次）

HPT 按 `train.total_iters` 控制步数，且 `epoch_size = len(train_loader)`（日志打印 `Epoch size: X`）。
先**单独跑一个客户端**看 `Epoch size`：

```bash
cd ~/HPT
/root/miniconda3/envs/hpt/bin/python -m hpt.run env=mujoco_metaworld \
  train.pretrained_dir=hf://liruiw/hpt-base dataset.episode_cnt=100 \
  domains=mujoco_metaworld_task1 +datasetsize=metaworld_1 \
  train.freeze_trunk=False train.total_iters=50 output_dir=output/calib \
  2>&1 | grep -E "Epoch size|Traj"
```

把 `LOCAL_TOTAL_ITERS` 改成 `3 × Epoch_size`（= 3 本地 epoch），填进 `fedavg_config.sh`。

---

## 6. 跑训练

> 前置：第 4 节 step 0/1/2 三个验证必须全过（trunk 键名、load_trunk、异构续训烟雾）。
> 推荐顺序：先 step 2 异构烟雾(1轮) → 再 (A) IID 健全性 → 最后 (B) 异构基线(50轮)。

```bash
cd ~/HPT
# (A) 先跑 IID 健全性：把 fedavg_config.sh 的 CLIENT_DOMAINS 全改成 task1
bash federated/run_fedavg.sh
# 期望：最终 S_fedavg(reach) ≈ 集中式 ~0.98。若差距大 → 管线有 bug，先修。

# (B) 异构基线：CLIENT_DOMAINS 改回 4 客户端分 3 任务（见 partition.py 输出）
bash federated/run_fedavg.sh
```

断点续训（某轮崩了从 r=10 接着跑）：`START_ROUND=10 bash federated/run_fedavg.sh`

> 远程若用 AutoDL：脚本已用**绝对路径** python 与输出目录，并用 `&&`/`;` 分隔；不要用 `cd ~/HPT nohup` 这种糊一行写法（会失败）。后台跑参考 `run_eval_fedavg.sh` 的 `nohup` 写法。

---

## 7. 评估 S_fedavg（与 S_central 同口径）

```bash
cd ~/HPT
GLOBAL_DIR=/root/HPT/output/fedavg_reach/global_r49 \
N_JOBS=1 \
  bash federated/run_eval_fedavg.sh
# 完成看：
grep -iE "success|rate|seed|mean|avg" /root/HPT/output/fedavg_reach/global_r49_eval.log | tail -n 30
```

**门 2 判据**：`S_fedavg`（异构）< `S_central`（0.98）且差距跳出 3 seeds 噪声带 → 复现"异构有害"，通过。
不过 → 把每客户端任务数降到 1、或加大异构（如 isolated 方案）。

---

## 8. 兜底方案（仅当 verify_continuation.py 返回 False）

`load_trunk` 本地若不可用，则**不依赖 HPT 的 `train.pretrained_dir` 续训**，改为自定义训练循环：
直接用 `Policy.from_pretrained(hf)` 初始化，每轮把全局 `model.pth` 用 `policy.load_state_dict(...)`
载入（覆盖 trunk），本地训 trunk 若干步，再 `policy.save(全局model.pth)`。聚合仍用 `aggregate.py`
（它读 `model.pth` 平均 trunk 即可）。自定义循环需要 `train_test.train` 的 forward/loss 接口，
届时按 `hpt/run.py` 里的训练体重写一个最小 step。

---

## 9. 后续扩展（同一 trunk 聚合接口）

- **H1（EDA 退化）**：用 `../fedvla_scaffold/fl_core.py` 的 `eda_weights` 替换 `aggregate.py` 的平均，
  跑均衡两簇划分，看相对 FedAvg 增益是否显著。
- **H2（按簇聚合）**：对 trunk 的专家选择向量做聚类，簇内聚合（`partition.py` 的 `isolated` 方案可造含孤立客户端的划分）。
- **MoE**：在 trunk 一半层加 top-k（k=2，8 专家），统计专家计数矩阵喂给 EDA（第 3 阶段）。
