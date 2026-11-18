#!/usr/bin/env bash
# 把第4阶段隐私侧的表格与图打包，供 pull_files.py 分块拉回本地。
set -u
OUT=/root/HPT/output/privacy
TGZ=/root/fedcore/priv_out.tgz
rm -f "$TGZ"
tar czf "$TGZ" -C "$OUT" \
  fig7_recon_action.png fig7_recon_curves.png fig7_image_feature.png fig7_sensitivity.png \
  privacy_table.md privacy_table.csv privacy_budget_table.md
echo "packed: $(stat -c %s $TGZ) bytes"
md5sum "$TGZ"
tar tzf "$TGZ"
