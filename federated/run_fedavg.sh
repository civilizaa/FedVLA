#!/usr/bin/env bash
# FedVLA 联邦训练编排（FedAvg）：4 客户端串行，50 轮，每轮本地训后聚合 trunk。
#
# 设计（已在 README 论证）：
#   - 复用 HPT 自带训练 `python -m hpt.run`，不重写训练循环。
#   - 第 0 轮各客户端从 INIT_PRETRAINED(hf) 训 3 epoch -> 各自 model.pth。
#   - 聚合：aggregate.py 平均 trunk -> 全局目录的 trunk.pth + model.pth + config*。
#   - 第 r+1 轮各客户端从全局目录续训（HPT init_policy 靠 trunk.pth 走本地续训分支）。
#   - 数据离线 Zarr 缓存，首轮落地后复用，不每轮重采。
#
# 用法：
#   bash run_fedavg.sh                # 从第 0 轮跑满 NUM_ROUNDS
#   START_ROUND=10 bash run_fedavg.sh  # 从指定轮续跑（断点续训）
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/fedavg_config.sh"

# ===== 环境（绝对路径 + 显式 export，避免新 shell 里 conda 失效）=====
export WANDB_MODE=offline
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

START_ROUND=${START_ROUND:-0}

echo "===== FedVLA FedAvg ====="
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "NUM_ROUNDS=$NUM_ROUNDS  NUM_CLIENTS=$NUM_CLIENTS  LOCAL_TOTAL_ITERS=$LOCAL_TOTAL_ITERS"
echo "CLIENT_DOMAINS: ${CLIENT_DOMAINS[*]}"
echo "从轮 $START_ROUND 开始"
echo "========================="

mkdir -p "$OUTPUT_ROOT"

for ((r=START_ROUND; r<NUM_ROUNDS; r++)); do
  echo "[round $r] 开始"

  # 全局目录（本轮聚合产物）
  GLOBAL_DIR="$OUTPUT_ROOT/global_r${r}"
  mkdir -p "$GLOBAL_DIR"

  # 上一轮全局目录（第 0 轮用 INIT_PRETRAINED）
  if [ "$r" -eq 0 ]; then
    PRETRAINED="$INIT_PRETRAINED"
  else
    PRETRAINED="$OUTPUT_ROOT/global_r$((r-1))"
  fi

  # ---- 客户端串行本地训练 ----
  for ((c=0; c<NUM_CLIENTS; c++)); do
    DOMAINS="${CLIENT_DOMAINS[$c]}"
    CLIENT_OUT="$OUTPUT_ROOT/client_${c}_r${r}"
    LOG="$OUTPUT_ROOT/log_client_${c}_r${r}.txt"
    echo "[round $r] client $c  domains=$DOMAINS  -> $CLIENT_OUT"

    "$HPT_PY" -m hpt.run \
      env=mujoco_metaworld \
      train.pretrained_dir="$PRETRAINED" \
      dataset.episode_cnt="$EPISODE_CNT" \
      domains="$DOMAINS" \
      +datasetsize="$DATASETSIZE" \
      train.total_iters="$LOCAL_TOTAL_ITERS" \
      output_dir="$CLIENT_OUT" \
      $EXTRA_HPT_ARGS \
      > "$LOG" 2>&1

    if [ ! -f "$CLIENT_OUT/model.pth" ]; then
      echo "[round $r] client $c 训练失败，查看 $LOG"; tail -n 30 "$LOG"; exit 1
    fi
  done

  # ---- 服务器端聚合 ----
  CLIENT_DIRS=()
  for ((c=0; c<NUM_CLIENTS; c++)); do CLIENT_DIRS+=("$OUTPUT_ROOT/client_${c}_r${r}"); done
  "$HPT_PY" "$SCRIPT_DIR/aggregate.py" \
    --clients "${CLIENT_DIRS[@]}" \
    --out "$GLOBAL_DIR" \
    >> "$OUTPUT_ROOT/log_aggregate_r${r}.txt" 2>&1

  if [ ! -f "$GLOBAL_DIR/trunk.pth" ]; then
    echo "[round $r] 聚合失败，查看 $OUTPUT_ROOT/log_aggregate_r${r}.txt"; exit 1
  fi
  echo "[round $r] 完成，全局模型 -> $GLOBAL_DIR"
done

echo "===== 全部 $NUM_ROUNDS 轮完成 ====="
echo "最终全局模型: $OUTPUT_ROOT/global_r$((NUM_ROUNDS-1))"
echo "下一步：用 federated/run_eval_fedavg.sh 评估（3 seeds × 20 rollouts）"
