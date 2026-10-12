# 环境踩坑日志

> 规则：每次遇到环境问题（装不上、跑不通、奇奇怪怪的报错），**先记在这里再解决**。
> 这是论文可复现性的资产，也是云实例被重置后能 20 分钟重建的依据。

---

## 已确认的事实 / 决策

### E1. 运行环境：云服务器（AutoDL），不用 VMware / WSL2
- **VMware Workstation 不支持把物理 GPU 暴露给虚拟机的 CUDA**（无 PCIe 直通，仅有虚拟显卡）。
  `nvidia-smi` / `torch.cuda.is_available()` 在 VMware Ubuntu 里都是 False，跑不了训练。
- 已充值 AutoDL，选 **RTX 3090 24GB** + PyTorch 2.x / Python 3.10 / CUDA 12.1 镜像。
- 大部分时间（装环境、写代码、改 bug）用**无卡模式**（约 0.1 元/时），只有训练切有卡。

### E2. Meta-World 不用单独装
- HPT 仓库内置完整副本：`env/mujoco/metaworld/`（含全部 50 个任务）。
- 环境名是 `-v2` 系列（如 `reach-v2`）。**千万不要** `pip install metaworld`（PyPI 3.x 用 gymnasium + MuJoCo 3，会冲突）。
- 验证：`python -c "from env.mujoco.metaworld.rollout_runner import ALL_TASK_CONFIG; print(len(ALL_TASK_CONFIG))"` → 50。

### E3. HPT 依赖用 `--no-deps` 安装
- 直接 `pip install -e .` 会连带拉 tensorflow 2.13 / rlds / zarr / dm-reverb / ray，重且易冲突。
- 用 `pip install -e . --no-deps` + 手动装子集（见 `setup_cloud.sh`）。

### E4. MuJoCo 需编译（Linux only）
- HPT 钉 `mujoco-py`（MuJoCo 2.1 的 Python 绑定），原生 Windows 走不通；云/Linux 正常。
- `MUJOCO_GL=egl` 用于离屏渲染；`WANDB_MODE=offline` 避免联网。

### E5. 两个已知脚本坑（第 1 步就会撞）
- `train.pretrained_dir` 会被拼成 `output/hf://...`，必须用命令行显式覆盖：
  `train.pretrained_dir=hf://liruiw/hpt-base`
- 评估脚本默认 `seed=range(3)` + `n_jobs=3`（并行 3 进程）。显存紧时先改 `n_jobs=1`。

### E6. 持久化红线
- 云系统盘不是你的硬盘，重置/过期会清空。
- 代码每天 push GitHub；checkpoint 放数据盘；results 每周五下载本地。

---

## 新坑记录模板（复制下面这段，填好再解决）

### [日期] E?: 现象一句话
- **环境**：(无卡/有卡, 镜像版本, GPU)
- **命令**：`...`
- **报错**：`...`（关键几行）
- **原因**：
- **解决**：
- **是否写进 setup_cloud.sh**：是 / 否（若是通用坑，应固化进脚本）

---
