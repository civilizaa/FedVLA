#!/usr/bin/env bash
# 精确时间账：P3-D 轮速 + 各组评估进度
set -u
OUT=/root/HPT/output
echo "=== 时间 ==="
date "+%F %T"
echo "=== p3_moe_iid 轮次时间戳 ==="
ls -ld --time-style=+%H:%M:%S $OUT/p3_moe_iid/global_r* 2>/dev/null | tail -8
echo "=== p3_moe_eda / cluster 完成时间 ==="
ls -ld --time-style=+%H:%M:%S $OUT/p3_moe_eda/global_r49 $OUT/p3_moe_eda_cluster/global_r49 $OUT/p3_moe_fedavg/global_r49 2>/dev/null
echo "=== 评估日志表数 ==="
for f in $OUT/eval_p3_*.log $OUT/eval_g2b_*.log; do
  [ -f "$f" ] || continue
  n=$(grep -c "| avg" "$f" 2>/dev/null || true); [ -n "$n" ] || n=0
  p=$(grep -oE "number of params \(M\): [0-9.]+" "$f" 2>/dev/null | head -1 | grep -oE "[0-9.]+$" || echo "-")
  echo "$(basename $f) tables=$n params=${p}M"
done
echo "=== 活跃评估 ==="
pgrep -af "hpt.run_eval" | cut -c1-95
echo "=== 活跃训练 ==="
pgrep -af "hpt.run env" | cut -c1-95
echo "=== 磁盘/GPU ==="
df -BG /root | tail -1
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
