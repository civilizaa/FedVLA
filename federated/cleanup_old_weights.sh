#!/usr/bin/env bash
# 磁盘清理：只删"已经彻底跑完"的实验组的历史轮次权重。
#
# ⚠️ 双重安全判据（缺一不可）：
#   1) 该组**没有**正在跑的 hpt.run 进程  → 不能删正在训练的组
#      （EDA 聚合每轮都要读 client_*_rN/expert_counts.pt，删了就崩）
#   2) 该组**没有**正在跑的 run_eval 进程 → 不能删正在评估的组
#      （评估要读 global_r49/model.pth）
#   3) 只删 global_r* 与 client_*_r* 目录，**日志一律保留**
#   4) 每组保留最近 KEEP 轮（默认 2）
#
# 用法：bash cleanup_old_weights.sh [KEEP]
set -u
OUT=/root/HPT/output
KEEP="${1:-2}"

echo "=== 清理前 ==="
df -BG /root | tail -1

for d in "$OUT"/*/; do
  tag=$(basename "$d")
  [ "$tag" = "eval" ] && continue

  # ---- 判据 1：正在训练？----
  if pgrep -af "hpt.run" 2>/dev/null | grep -q "$tag"; then
    echo "[SKIP] $tag 正在训练，跳过"
    continue
  fi
  # ---- 判据 2：正在评估？----
  if pgrep -af "run_eval" 2>/dev/null | grep -q "$tag"; then
    echo "[SKIP] $tag 正在评估，跳过"
    continue
  fi

  # ---- 收集该组的历史轮次 ----
  rounds=$(ls -d "$d"global_r* 2>/dev/null | sed 's/.*global_r//' | sort -n)
  n=$(echo "$rounds" | grep -c '[0-9]' || true)
  [ -n "$n" ] || n=0
  if [ "$n" -le "$KEEP" ]; then
    continue
  fi

  # ---- 删掉超出 KEEP 的历史轮次（权重目录，日志保留）----
  keep_from=$(echo "$rounds" | tail -n "$KEEP" | head -1)
  freed=0
  for r in $rounds; do
    [ "$r" -lt "$keep_from" ] || continue
    for sub in global_r$r client_*_r$r; do
      for p in "$d$sub"; do
        [ -d "$p" ] || continue
        sz=$(du -sm "$p" 2>/dev/null | cut -f1)
        [ -n "$sz" ] || sz=0
        rm -rf "$p"
        freed=$((freed + sz))
      done
    done
  done
  echo "[CLEAN] $tag 释放 ${freed}MB（保留 r>=${keep_from}）"
done

echo
echo "=== 清理后 ==="
df -BG /root | tail -1
