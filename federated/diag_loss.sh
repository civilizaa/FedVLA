#!/usr/bin/env bash
# 抓各实验组**最后一轮**每个客户端的训练 loss，判断训练是否收敛 / 是否崩坏。
# 训练日志路径：$OUT/<tag>/log_client_<c>_r<r>.txt
set -u
OUT=/root/HPT/output

for tag in p3_moe_fedavg p3_moe_eda p3_moe_eda_cluster p3_moe_iid g2b_hetero g2b_iid_e4; do
  d="$OUT/$tag"
  [ -d "$d" ] || continue
  echo "=== $tag ==="
  # 找该组最大的轮次号
  last=$(ls "$d"/log_client_*_r*.txt 2>/dev/null | sed 's/.*_r\([0-9]*\)\.txt/\1/' | sort -n | tail -1)
  [ -n "$last" ] || { echo "  无训练日志"; continue; }
  for f in "$d"/log_client_*_r${last}.txt; do
    [ -f "$f" ] || continue
    c=$(basename "$f" | sed 's/log_client_\([0-9]*\)_r.*/\1/')
    # 抓 loss 行（多种写法都试）
    l=$(grep -oiE "loss[^0-9a-z]{0,3}[0-9]+\.?[0-9]*(e-?[0-9]+)?" "$f" | tail -1 \
        | grep -oE "[0-9]+\.?[0-9]*(e-?[0-9]+)?" | tail -1)
    # 也抓 train/val 明细里的最后一行
    tail_line=$(grep -E "Epoch size:|train.*loss|val.*loss" "$f" | tail -2 | tr '\n' ' ')
    printf "  c%s r%-3s loss=%-12s | %s\n" "$c" "$last" "${l:-?}" "$tail_line"
  done
  echo
done
echo "LOSSDIAG_DONE"
