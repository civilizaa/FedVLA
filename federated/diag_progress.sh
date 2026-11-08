#!/usr/bin/env bash
# P3-D 训练进度 + 磁盘占用分布 + 评估守护状态
set -u
OUT=/root/HPT/output
echo "=== 时间 ==="
date "+%F %T"
echo "=== p3_moe_iid 轮次 ==="
ls -ld --time-style=+%H:%M:%S $OUT/p3_moe_iid/global_r* 2>/dev/null | tail -6
echo "=== p3_moe_iid client dirs ==="
ls -d $OUT/p3_moe_iid/client_*_r* 2>/dev/null | tail -6
echo "=== 磁盘 ==="
df -BG /root | tail -1
echo "=== 各组占用 (MB) ==="
for d in $OUT/*/; do
  tag=$(basename "$d")
  sz=$(du -sm "$d" 2>/dev/null | cut -f1)
  echo "$tag ${sz}MB"
done
echo "=== 权重目录明细 (p3_moe_iid) ==="
for g in $OUT/p3_moe_iid/global_r*/; do
  [ -d "$g" ] || continue
  echo "$(basename $g): $(ls $g 2>/dev/null | tr '\n' ' ')"
done
echo "=== opt_state ==="
ls -la $OUT/*/opt_state_c*.pt 2>/dev/null | awk '{print $5, $9}'
echo "=== 评估守护 ==="
pgrep -af "phase3_eval_watch|phase3_early_eval|gate2b_eval_watch" | cut -c1-80
echo "--- phase3_eval_watch.log ---"
tail -n 6 $OUT/phase3_eval_watch.log 2>/dev/null
echo "=== PHASE3_DONE 标记 ==="
grep -rn "PHASE3_DONE" $OUT/*.log 2>/dev/null | tail -5
