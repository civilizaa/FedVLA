#!/usr/bin/env bash
# 查提前评估的 MoE 自验证结果（参数量断言）。
set -u
LOG=/root/HPT/output/eval_p3_moe_fedavg_mujoco_metaworld_task1.log
echo "=== early eval log ==="
tail -n 12 /root/HPT/output/phase3_early_eval.log 2>/dev/null
echo "=== p3_moe_fedavg task1 主日志 ==="
if [ -f "$LOG" ]; then
  echo "size=$(stat -c %s "$LOG")"
  echo "--- params ---"
  grep -oE "number of params \(M\): [0-9.]+" "$LOG" | head -3
  echo "--- avg tables ---"
  n=$(grep -c "| avg" "$LOG" 2>/dev/null || true); [ -n "$n" ] || n=0
  echo "tables=$n"
else
  echo "主日志尚未生成"
fi
echo "=== 训练轮次 ==="
ls -ld --time-style=+%H:%M:%S /root/HPT/output/p3_moe_eda/global_r* 2>/dev/null
echo "=== 磁盘 ==="
df -BG /root | tail -1
