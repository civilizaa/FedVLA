#!/usr/bin/env bash
# FedVLA 云端前置检查：一键拉代码 + 门0 + 门1
#
# 用法（在 AutoDL 的 ~/HPT 下执行）：
#     bash /root/HPT/federated/cloud_gates.sh
# 或者从本机 scp 上来后执行。
#
# 门0 / 门1 都是纯 CPU 操作，**可以在无卡模式跑**（省 ¥0.9/小时）。
# 门2（异构续训烟雾）必须开卡，见 run_fedavg.sh。
#
# 设计：脚本不依赖 git 能用——代码到位检测失败时会自动尝试从 raw.githubusercontent 补齐。
set -u

HPT_PY=/root/miniconda3/envs/hpt/bin/python
HPT_DIR=/root/HPT
FEDVLA_DIR=/root/FedVLA
RAW_BASE="https://raw.githubusercontent.com/hp/FedVLA/master"

export MUJOCO_GL=egl
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export PYTHONPATH="$HPT_DIR:${PYTHONPATH:-}"

cd "$HPT_DIR" || { echo "找不到 $HPT_DIR"; exit 1; }

echo "################################################"
echo "# 步骤 1/3：确保 federated/ 代码到位"
echo "################################################"
if [ -f "$FEDVLA_DIR/federated/aggregate.py" ]; then
  echo "[OK] federated/ 已存在，跳过下载"
else
  echo "[..] federated/ 缺失，尝试补齐"
  if [ ! -d "$FEDVLA_DIR" ]; then
    echo "[..] $FEDVLA_DIR 不存在，先 git clone（国内可能很慢）"
    git clone --depth 1 https://github.com/hp/FedVLA.git "$FEDVLA_DIR" 2>&1 | tail -5
  fi
  mkdir -p "$FEDVLA_DIR/federated"
  for f in aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md; do
    echo "  -> $f"
    curl -sSL --max-time 60 -o "$FEDVLA_DIR/federated/$f" "$RAW_BASE/federated/$f" \
      || echo "     下载失败：$f"
  done
  # 校验：aggregate.py 至少要拿到
  if [ -f "$FEDVLA_DIR/federated/aggregate.py" ]; then
    echo "[OK] federated/ 补齐完成"
  else
    echo "[!!] 自动补齐失败。请手动用 scp 传 federated/ 目录，或用国内镜像重试 git clone"
    exit 1
  fi
fi
ls -la "$FEDVLA_DIR/federated/"

echo
echo "################################################"
echo "# 步骤 2/3：门0 —— 打印 model.pth 的 trunk 键名"
echo "################################################"
"$HPT_PY" - <<'PY'
import os, torch
from collections import Counter

p = "output/reach_baseline/model.pth"
if not os.path.exists(p):
    print(f"[!!] 找不到 {p}（需先完成门1的集中式训练）")
    raise SystemExit(0)

sd = torch.load(p, map_location="cpu", weights_only=False)
print("state_dict 总键数:", len(sd))
print("顶层前缀分布:", Counter(k.split(".")[0] for k in sd).most_common(12))

tk = [k for k in sd if k.startswith("trunk")]
print()
print("trunk 键数量:", len(tk))
print("trunk 键示例:", tk[:6])

# 判定 aggregate.py 的假设是否成立
if not tk:
    print()
    print("[!!] 没有任何以 'trunk.' 开头的键！")
    print("     => aggregate.py 的 _is_trunk_key 会漏选，必须改判定逻辑。")
    print("     请把上面的『顶层前缀分布』贴给 AI 助手。")
else:
    inner = tk[0][len("trunk."):]
    print()
    print("[OK] trunk 键以 'trunk.' 开头 => aggregate.py 假设成立")
    print("     去掉首层 'trunk.' 后示例键:", inner)
    print("     (这正是写进 trunk.pth 的键名，供 policy.load_trunk 使用)")
PY

echo
echo "################################################"
echo "# 步骤 3/3：门1 —— 验证本地 trunk.pth 续训"
echo "################################################"
"$HPT_PY" "$FEDVLA_DIR/federated/verify_continuation.py"
echo
echo "门1 脚本退出码: $?"
echo
echo "================================================"
echo "若门0 显示 [OK] 且门1 输出 True -> 续训链路成立"
echo "下一步：开卡，改 fedavg_config.sh 为 NUM_ROUNDS=1 做门2 异构续训烟雾"
echo "================================================"
