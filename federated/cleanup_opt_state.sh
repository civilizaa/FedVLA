#!/usr/bin/env bash
# 清理已完成实验组的 Adam 优化器状态（opt_state_c*.pt）。
#
# 安全依据：
#   - opt_state_c*.pt 只在**该组还在训练时**被 run_fedavg.sh 读/写
#     （FEDVLA_OPT_STATE="$OUTPUT_ROOT/opt_state_c${c}.pt"）
#   - 组一旦训练结束，这个文件再也不会被读
#   - 评估只读 model.pth/trunk.pth，**不碰 opt_state** → 清理不影响评估
#
# ⚠️ 关键判据：必须已到**最后一轮**（global_r49/trunk.pth 存在）。
#   只满足"有任一完整轮次"是不够的 —— 那种情况可能是**中断**的组，
#   续跑时需要 Adam 状态（删了会丢失动量，实验口径就变了）。
#   只有确认跑满 50 轮（r49 存在）才敢删。
#
# 用法：bash cleanup_opt_state.sh
set -u
OUT=/root/HPT/output
NUM_ROUNDS="${NUM_ROUNDS:-50}"
freed_total=0

for d in "$OUT"/*/; do
  tag=$(basename "$d")
  # 只处理有 opt_state 的组
  ls "$d"opt_state_c*.pt >/dev/null 2>&1 || continue
  # 该组正在训练？→ 绝不能删
  if pgrep -af "hpt.run" 2>/dev/null | grep -q "$tag"; then
    echo "[SKIP] $tag 正在训练，保留 opt_state"
    continue
  fi
  # ⚠️ 必须已跑满最后一轮才敢删（中断的组要留着续跑用）
  if [ ! -f "$d/global_r$((NUM_ROUNDS - 1))/trunk.pth" ]; then
    echo "[SKIP] $tag 未跑满 r$((NUM_ROUNDS - 1))（可能是中断），保留 opt_state"
    continue
  fi
  freed=0
  for f in "$d"opt_state_c*.pt; do
    [ -f "$f" ] || continue
    sz=$(du -sm "$f" | cut -f1); [ -n "$sz" ] || sz=0
    rm -f "$f"
    freed=$((freed + sz))
  done
  echo "[CLEAN] $tag 释放 ${freed}MB（opt_state，已跑满 r$((NUM_ROUNDS - 1))）"
  freed_total=$((freed_total + freed))
done

echo
echo "总计释放 ${freed_total}MB"
df -BG /root | tail -1
