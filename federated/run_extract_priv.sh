#!/usr/bin/env bash
# 第4阶段隐私侧：重新提取结果（modality_rev 补跑完成后必须重跑）。
set -u
PY=/root/miniconda3/envs/hpt/bin/python
OUT=/root/HPT/output/privacy
"$PY" /root/fedcore/extract_privacy_results.py \
  --json "$OUT/privacy_results.json" \
  --out "$OUT"
echo "extract rc=$?"
echo "--- 输出目录 ---"
ls -la "$OUT"
