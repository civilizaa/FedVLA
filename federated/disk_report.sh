#!/usr/bin/env bash
# 磁盘占用分析（避开 WAF：命令写成脚本再执行）。
set -u
echo "=== output/ 顶层占用 TOP 14 ==="
du -sh /root/HPT/output/* 2>/dev/null | sort -rh | head -14
echo
echo "=== 各 p3 实验组占用 ==="
for t in p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster p3_moe_iid; do
  printf "%-22s %s  rounds=%s\n" "$t" \
    "$(du -sh /root/HPT/output/$t 2>/dev/null | cut -f1)" \
    "$(ls -d /root/HPT/output/$t/global_r* 2>/dev/null | wc -l)"
done
echo
echo "=== client_* 目录数（每组每轮 4 个客户端产物）==="
for t in p3_moe_fedavg p3_moe_eda; do
  echo "$t client_dirs=$(ls -d /root/HPT/output/$t/client_* 2>/dev/null | wc -l)"
done
echo
echo "=== 磁盘 ==="
df -BG /root | tail -1
