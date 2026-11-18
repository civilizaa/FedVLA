#!/usr/bin/env bash
# FedVLA 第4阶段：DP 效用扫描的 **C 标定**（跑 2 轮干净联邦训练，测真实更新范数）。
#
# 为什么需要标定 C：
#   DP-FedAvg 的加噪形式是  Δ·min(1, C/‖Δ‖₂) + N(0, (σ·C)²I)。
#   隐私参数只有 σ（噪声倍率 = σ·C / C = σ）⇒ **ε 只依赖 σ**；
#   C 只决定效用，不决定隐私。所以 C 可以按数据自由选取，
#   惯例取"典型更新范数" ⇒ 此时 σ 直接读作 **噪声/信号比**。
#
#   ⚠️ 隐私侧的 σ 网格（1e-4 ~ 1）是针对"单位范数梯度"定义的，
#      真实训练里的参数更新范数量级完全不同 ⇒ 不能照搬，
#      必须先标定 C，再用 σ ∈ {0.01, 0.1, 1.0} 这种"噪声/信号比"语义。
#
# 本脚本只跑 2 轮（r=0 初始化 + r=1 产生第一个真实更新量），
# 训练全程**不加噪**（DP_CALIBRATE=1 ⇒ 钩子只测量不写回）。
#
# 幂等：已存在 calib.json 则直接退出。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${OUT:-/root/HPT/output}
CALIB_DIR="$OUT/dp_calib"
HPT_PY=/root/miniconda3/envs/hpt/bin/python

if [ -f "$CALIB_DIR/calib.json" ]; then
  echo "[标定] 已存在 $CALIB_DIR/calib.json，跳过"
  cat "$CALIB_DIR/calib.json"
  exit 0
fi

# ===== MoE 环境变量（与 P3-C / 通信扫描逐字一致）=====
export FEDVLA_MOE=1
export FEDVLA_MOE_EXPERTS=8
export FEDVLA_MOE_TOPK=2
export FEDVLA_MOE_EXPERT_RATIO=0.25
export FEDVLA_MOE_LAYERS=half

# ===== 异构划分（与 P3-C 完全一致）=====
HETERO_DOMAINS="mujoco_metaworld_task1 mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1"
HETERO_ENVS="reach-v2 push-v2 button-press-topdown-v2 reach-v2"
HETERO_EPOCH_SIZE="15 19 24 15"

mkdir -p "$CALIB_DIR"
rm -f "$CALIB_DIR/delta_norm.jsonl"

echo "===== DP 效用扫描：C 标定（2 轮，干净训练）====="
echo "OUTPUT_ROOT=$CALIB_DIR"

OUTPUT_ROOT="$CALIB_DIR" NUM_ROUNDS=2 \
CLIENT_DOMAINS_OVERRIDE="$HETERO_DOMAINS" \
CLIENT_ENVS_OVERRIDE="$HETERO_ENVS" \
CLIENT_EPOCH_SIZE_OVERRIDE="$HETERO_EPOCH_SIZE" \
USE_MOE=1 AGG_MODE=eda_cluster \
KEEP_ROUNDS=1 MIN_FREE_GB=4 \
POST_CLIENT_HOOK=dp_noise_client.py \
DP_CALIBRATE=1 DP_STAT_OUT="$CALIB_DIR/delta_norm.jsonl" \
  bash "$SCRIPT_DIR/run_fedavg.sh" > "$CALIB_DIR/calib_train.log" 2>&1

rc=$?
if [ "$rc" != "0" ]; then
  echo "[!!] 标定训练失败（rc=$rc），查看 $CALIB_DIR/calib_train.log"
  tail -n 40 "$CALIB_DIR/calib_train.log"
  exit 1
fi

if [ ! -s "$CALIB_DIR/delta_norm.jsonl" ]; then
  echo "[!!] 没有采到任何更新范数 —— 钩子可能没被调用"
  tail -n 30 "$CALIB_DIR/calib_train.log"
  exit 1
fi

"$HPT_PY" "$SCRIPT_DIR/dp_clip_calibrate.py" \
  --stat "$CALIB_DIR/delta_norm.jsonl" --out "$CALIB_DIR/calib.json"
rc=$?
if [ "$rc" != "0" ]; then
  echo "[!!] 标定汇总失败（rc=$rc）"
  exit 1
fi

echo "===== 标定完成 ====="
cat "$CALIB_DIR/calib.json"
