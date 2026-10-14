#!/usr/bin/env bash
# ============================================================
# 云服务器一键环境安装（HPT + Meta-World + 联邦 VLA 实验）
# 适用：AutoDL / 恒源云 等 GPU 云实例，Ubuntu 20.04 / 22.04
# 用法：bash setup_cloud.sh
# 幂等：可重复执行，已完成的步骤会跳过
# ============================================================
set -e

# ---- GitHub 镜像代理（国内云实例直连 github.com 常超时）----
# 用 gitclone.com 做代理；先清掉可能残留的失效代理（ghproxy.com 等），避免多条 insteadOf 冲突
git config --global --unset-all url."https://ghproxy.com/https://github.com/".insteadOf 2>/dev/null || true
git config --global --unset-all url."https://ghproxy.net/https://github.com/".insteadOf 2>/dev/null || true
git config --global --unset-all url."https://github.com.cnpmjs.org/".insteadOf 2>/dev/null || true
git config --global url."https://gitclone.com/github.com/".insteadOf "https://github.com/"

echo "==> [1/7] 系统依赖"
if ! command -v patchelf >/dev/null 2>&1; then
  apt-get update
  apt-get install -y \
    build-essential \
    libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
    libegl1-mesa libegl1-mesa-dev libosmesa6-dev \
    libxrender1 libxext6 libsm6 \
    patchelf wget curl git unzip
else
  echo "    已安装，跳过"
fi

echo "==> [2/7] MuJoCo 2.1.0"
if [ ! -d "$HOME/.mujoco/mujoco210" ]; then
  mkdir -p "$HOME/.mujoco"
  cd "$HOME/.mujoco"
  wget "https://ghproxy.net/https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz" -O mujoco210.tar.gz --no-check-certificate
  tar -xvzf mujoco210.tar.gz
  cd - >/dev/null
else
  echo "    已存在，跳过"
fi

echo "==> [3/7] 环境变量写入 ~/.bashrc"
if ! grep -q "FEDVLA_ENV_SETUP" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<'EOF'

# ---- FEDVLA_ENV_SETUP ----
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
# --------------------------
EOF
  echo "    已写入（重新登录后生效，当前 shell 请执行 source ~/.bashrc）"
else
  echo "    已写入，跳过"
fi
source "$HOME/.bashrc" 2>/dev/null || true
# 显式导出（bashrc 在非交互 shell 下会因开头的 return 而未生效）
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export MUJOCO_GL=egl
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com

echo "==> [4/7] conda 环境（Python 3.10）"
if ! command -v conda >/dev/null 2>&1; then
  echo "    !! 未检测到 conda。若镜像自带 conda，请先 source 对应路径，例如："
  echo "       source /opt/conda/etc/profile.d/conda.sh"
  echo "    或装 miniconda：https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^hpt "; then
  echo "    环境 hpt 已存在，跳过创建"
else
  conda create -n hpt python=3.10 -y
fi
conda activate hpt

echo "==> [5/7] PyTorch（2.1.1，从 PyPI 装，自带 CUDA 12.1 运行时）"
# download.pytorch.org 在国内云实例常被 403，改用 PyPI 默认源（pypi.org 已验证可达）。
# PyPI 上的 torch 2.1.1 Linux wheel 即为 CUDA 构建；装完自检会确认 cuda.is_available()。
if python -c "import torch; assert torch.__version__.startswith('2.1.1')" 2>/dev/null; then
  echo "    已存在 torch 2.1.1，跳过"
else
  pip install torch==2.1.1 torchvision==0.16.1
fi

echo "==> [6/7] HPT 与其依赖子集（跳过 tensorflow/rlds/dm-reverb/zarr/ray）"
HPT_DIR="${HPT_DIR:-$HOME/HPT}"
if [ ! -d "$HPT_DIR" ]; then
  git clone https://github.com/liruiw/HPT.git "$HPT_DIR"
fi
cd "$HPT_DIR"
pip install -e . --no-deps
pip install "numpy==1.24.3" "gym==0.26.2" "gymnasium>=0.29.0" \
  "mujoco-py" "Cython==0.29.36" "scipy==1.10.1" "imageio==2.27.0" \
  "hydra-core==1.3.2" "hydra-joblib-launcher" "einops==0.7.0" "timm==0.9.12" \
  "transformers==4.36.0" "scikit-learn==1.3.2" "matplotlib==3.7.4" \
  "seaborn==0.13.0" "opencv-python==4.8.1.78" "tqdm==4.66.1" "tabulate==0.9.0" \
  "easydict==1.11" "transforms3d==0.4.1" "wandb==0.16.1" "lxml==4.9.3" \
  "GPUtil==1.4.0" "pyvirtualdisplay==3.0" "protobuf==3.20.3"

echo "==> [7/7] 自检"
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "mujoco_py", "hpt", "numpy", "hydra", "einops"]:
    try:
        importlib.import_module(m)
        print(f"    [ok] {m}")
    except Exception as e:
        ok = False
        print(f"    [FAIL] {m}: {e}")
try:
    import torch
    print(f"    [{'ok' if torch.cuda.is_available() else 'FAIL'}] torch.cuda.is_available() = {torch.cuda.is_available()}")
except Exception:
    pass
sys.exit(0 if ok else 1)
PY

cat <<'EOF'

============================================================
安装完成。三件事：

1. Meta-World 已经内置在 HPT 仓库里（env/mujoco/metaworld/），
   **不要 pip install metaworld** —— 装了反而可能和内置版本冲突。
   验证（在 HPT 目录下执行，应输出 50）：
   python -c "from env.mujoco.metaworld.rollout_runner import ALL_TASK_CONFIG; print(len(ALL_TASK_CONFIG))"

2. 环境名是 -v2 系列（reach-v2 / push-v2 / button-press-topdown-v2 / door-open-v2），
   用的是老 gym API，所以 gym==0.26.2 必须装（脚本已装）。

3. 第一次训练冒烟：
   python -m hpt.run env=mujoco_metaworld domains=mujoco_metaworld_task1 \
     +datasetsize=metaworld_1 dataset.episode_cnt=100 \
     train.pretrained_dir=hf://liruiw/hpt-base +mode=debug
============================================================
EOF
