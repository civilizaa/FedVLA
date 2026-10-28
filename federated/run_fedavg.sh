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

# ===== 开训前硬拦截：load_trunk 补丁必须存在 =====
# 不打补丁的后果是"静默"的：客户端每轮都从 hf 原始权重重新开始，
# 联邦退化成"每轮独立训练"，聚合的权重根本没被用上 —— 不报错，结果全废。
# 所以在烧 50 轮算力之前必须拦下来。
POLICY_FILE="${HPT_DIR:-/root/HPT}/hpt/models/policy.py"
if [ ! -f "$POLICY_FILE" ]; then
  echo "[!!] 找不到 $POLICY_FILE"
  exit 1
fi
if ! grep -q "FedVLA patch" "$POLICY_FILE"; then
  echo "[!!] 检测到 HPT 的 load_trunk 未打补丁 —— 联邦续训会静默失效！"
  echo "     请先执行：bash $SCRIPT_DIR/patch_inline.sh"
  echo "     （或 python $SCRIPT_DIR/patch_hpt_load_trunk.py）"
  exit 1
fi
echo "[OK] load_trunk 补丁已就位"

# ===== 开训前硬拦截：全局目录必须有 extras.pth =====
# extras.pth 存 modalities_tokens（policy 顶层 ParameterDict，不在 trunk 内）。
# 缺失时续训的模态 token 保持随机初始化，共享 trunk 会收到不一致输入。
for ((rr=1; rr<NUM_ROUNDS; rr++)); do
  GD="$OUTPUT_ROOT/global_r$((rr-1))"
  if [ -d "$GD" ] && [ -f "$GD/trunk.pth" ] && [ ! -f "$GD/extras.pth" ]; then
    echo "[WARN] $GD 缺 extras.pth（该轮聚合是老版本产物）—— 模态 token 将随机初始化"
  fi
done

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

  # ---- 续训正确性自检（只对第 1 轮及以后做）----
  # 目的：证明客户端模型里的 trunk 与上一轮全局模型**确实不同**，
  #      即聚合的权重真的被载入并继续训练了。
  #      若相同，说明续训没生效（load_trunk 空操作 / 权重没被覆盖）→ 立刻停。
  if [ "$r" -ge 1 ]; then
    PREV_GLOBAL="$OUTPUT_ROOT/global_r$((r-1))"
    if [ -f "$PREV_GLOBAL/trunk.pth" ]; then
      "$HPT_PY" - "$PREV_GLOBAL/trunk.pth" "$CLIENT_OUT/model.pth" <<'PY' || exit 1
import sys, torch
prev_path, cur_path = sys.argv[1], sys.argv[2]
prev = torch.load(prev_path, map_location="cpu", weights_only=False)
cur = torch.load(cur_path, map_location="cpu", weights_only=False)
# 全局 trunk.pth 的键是 'trunk.xxx'；完整模型里对应 'trunk.trunk.xxx'
diffs = []
for k, v in prev.items():
    ck = "trunk." + k
    if ck in cur:
        diffs.append((float((cur[ck].float() - v.float()).abs().max()), k))
if not diffs:
    print("[!!] 无法比对 trunk 键（键名不匹配）")
    sys.exit(1)
mx = max(diffs)
print(f"[续训自检] 客户端 trunk 与上一轮全局的最大差异 = {mx[0]:.3e} @ {mx[1]}")
if mx[0] == 0.0:
    print("[!!] 客户端 trunk 与上一轮全局完全相同 —— 聚合权重未被载入，联邦是空转！")
    sys.exit(1)
print("[OK] 续训生效：客户端确实在上一轮全局权重基础上继续训练")
PY
    fi
  fi

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
