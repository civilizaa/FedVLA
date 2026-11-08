#!/usr/bin/env bash
# CPU 负载诊断：列出占用最高的进程，定位是谁在抢核。
set -u
echo "=== uptime ==="
uptime
echo "=== nproc ==="
nproc
echo "=== top cpu procs ==="
ps -eo pcpu,pid,cmd --sort=-pcpu | head -14 | cut -c1-110
echo "=== eval procs ==="
pgrep -af "hpt.run_eval" | cut -c1-100
echo "=== train procs ==="
pgrep -af "hpt.run env" | cut -c1-100
echo "=== threads per proc ==="
for p in $(pgrep -f "hpt.run_eval|hpt.run env" | head -8); do
  echo "pid=$p threads=$(ls /proc/$p/task 2>/dev/null | wc -l) cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-70)"
done
echo "=== p3 rounds ==="
ls -ld --time-style=+%H:%M:%S /root/HPT/output/p3_moe_eda/global_r* 2>/dev/null
