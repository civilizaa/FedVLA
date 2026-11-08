#!/usr/bin/env bash
# 第 3 阶段总状态快照。
set -u
OUT=/root/HPT/output
echo "时间: $(date '+%F %T')"
echo
echo "=== 各实验组进度 ==="
for t in g2b_iid_v1 g2b_iid_v2 g2b_iid_e4 g2b_hetero p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster p3_moe_iid; do
  d="$OUT/$t"
  [ -d "$d" ] || continue
  last=$(ls -d "$d"global_r* 2>/dev/null | sed 's/.*global_r//' | sort -n | tail -1)
  n=$(ls -d "$d"global_r* 2>/dev/null | wc -l)
  done_ok="N"
  [ -f "$d/global_r49/trunk.pth" ] && done_ok="Y"
  printf "  %-20s last_r=%-4s dirs=%-3s r49_done=%s\n" "$t" "${last:-?}" "$n" "$done_ok"
done
echo
echo "=== 评估日志（表数 >= 3 为完成）==="
for f in "$OUT"/eval_g2b_*.log "$OUT"/eval_p3_*.log; do
  [ -f "$f" ] || continue
  n=$(grep -c "| avg" "$f" 2>/dev/null || true)
  [ -n "$n" ] || n=0
  printf "  %-52s tables=%s\n" "$(basename "$f")" "$n"
done
echo
echo "=== 活跃进程 ==="
pgrep -af "hpt.run" 2>/dev/null | grep -v "pgrep" | cut -c1-110
echo
echo "=== 磁盘 / GPU ==="
df -BG /root | tail -1
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
