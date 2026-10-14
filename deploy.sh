#!/usr/bin/env bash
# ============================================================
# FedVLA 一键部署（解压就位 + 安装 + 验证）
# 适用：AutoDL 实例，已用 JupyterLab 上传了以下 zip 到 $HOME：
#   - fedvla-repo.zip   （本文件所在仓库的压缩包，含 setup_cloud.sh）
#   - HPT-main.zip      （从 github.com/liruiw/HPT 点 Code -> Download ZIP）
# 用法（在容器 Terminal 里）：
#   cd ~ && bash deploy.sh
# ============================================================
set -e

HD="$HOME"
cd "$HD"

# ---------- 1. 就位 FedVLA ----------
if [ -f "$HD/FedVLA/setup_cloud.sh" ]; then
  echo "[deploy] FedVLA 已就位，跳过解压"
elif [ -f "$HD/fedvla-repo/setup_cloud.sh" ]; then
  mv "$HD/fedvla-repo" "$HD/FedVLA"
  echo "[deploy] 已将 fedvla-repo 改名为 FedVLA"
else
  for z in "$HD/FedVLA.zip" "$HD/fedvla-repo.zip"; do
    if [ -f "$z" ]; then
      echo "[deploy] 解压 $z"
      unzip -o "$z" -d "$HD"
      if [ -d "$HD/fedvla-repo" ] && [ ! -d "$HD/FedVLA" ]; then
        mv "$HD/fedvla-repo" "$HD/FedVLA"
      fi
      break
    fi
  done
fi

if [ ! -f "$HD/FedVLA/setup_cloud.sh" ]; then
  echo "!! 找不到 $HD/FedVLA/setup_cloud.sh"
  echo "   请确认已把 fedvla-repo.zip 上传到 $HD（JupyterLab 左侧上传按钮）"
  exit 1
fi

# ---------- 2. 就位 HPT ----------
if [ -d "$HD/HPT" ]; then
  echo "[deploy] HPT 已就位，跳过解压"
else
  for z in "$HD/HPT.zip" "$HD/HPT-main.zip" "$HD/HPT-master.zip"; do
    if [ -f "$z" ]; then
      echo "[deploy] 解压 $z"
      unzip -o "$z" -d "$HD"
      if [ -d "$HD/HPT-main" ] && [ ! -d "$HD/HPT" ]; then
        mv "$HD/HPT-main" "$HD/HPT"
      elif [ -d "$HD/HPT-master" ] && [ ! -d "$HD/HPT" ]; then
        mv "$HD/HPT-master" "$HD/HPT"
      fi
      break
    fi
  done
fi

if [ ! -d "$HD/HPT" ]; then
  echo "!! 找不到 $HD/HPT，setup_cloud.sh 将尝试 git clone（需要网络，可能失败）"
  echo "   推荐：从 github.com/liruiw/HPT 下载 ZIP 并上传为 HPT-main.zip 后重跑本脚本"
fi

# ---------- 3. 安装 ----------
cd "$HD/FedVLA"
source "$HD/.bashrc" 2>/dev/null || true
bash setup_cloud.sh

# ---------- 4. 验证 ----------
echo "==> [deploy] 验证 Meta-World（正常应输出 50）"
cd "$HD/HPT"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate hpt 2>/dev/null || true
python -c "from env.mujoco.metaworld.rollout_runner import ALL_TASK_CONFIG; print('TASKS=', len(ALL_TASK_CONFIG))" \
  || echo "!! 验证未通过，请把上面安装阶段的输出贴给助手排查"

echo "============================================================"
echo "[deploy] 完成。若上面打印 TASKS= 50，环境即装好。"
echo "  - 不训练时请到 AutoDL 控制台关机（不是销毁）。"
echo "  - 之后要跑第一条训练冒烟，在 3090 模式下执行："
echo "      conda activate hpt && cd ~/HPT"
echo "      python -m hpt.run env=mujoco_metaworld domains=mujoco_metaworld_task1 \\"
echo "        +datasetsize=metaworld_1 dataset.episode_cnt=100 \\"
echo "        train.pretrained_dir=hf://liruiw/hpt-base +mode=debug"
echo "============================================================"
