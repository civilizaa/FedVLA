#!/usr/bin/env bash
# 细查单个实验组的磁盘占用结构。
set -u
for tag in g2b_iid_e4 g2b_hetero fedavg_iid p3_moe_fedavg; do
  d="/root/HPT/output/$tag"
  [ -d "$d" ] || continue
  echo "=== $tag  总计 $(du -sh "$d" | cut -f1) ==="
  echo "  global_r* 目录:"
  ls -d "$d"global_r* 2>/dev/null | while read -r g; do
    echo "    $(basename "$g") $(du -sh "$g" | cut -f1)"
  done
  echo "  client_* 目录数: $(ls -d "$d"client_* 2>/dev/null | wc -l)"
  echo "  其他文件:"
  find "$d" -maxdepth 1 -type f 2>/dev/null | head -3 | while read -r f; do
    echo "    $(basename "$f") $(du -sh "$f" | cut -f1)"
  done
  echo "  client 单个目录大小: $(du -sh "$d"client_* 2>/dev/null | head -1 | cut -f1)"
  echo
done
df -BG /root | tail -1
