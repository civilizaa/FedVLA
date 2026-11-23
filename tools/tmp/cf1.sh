set -u
K=/root/fedcore/keep
# 1) 备份 DP 的 client_2（task3 干净 normalizer 来源；DP 收尾也会被删）
if [ ! -d "$K/dp_client2_r1" ] && [ -d /root/HPT/output/dp_s0001/client_2_r1 ]; then
  cp -r /root/HPT/output/dp_s0001/client_2_r1 "$K/dp_client2_r1" && echo "[OK] 备份 dp_client2_r1"
fi
ls -la $K/dp_client2_r1/ 2>/dev/null | head -6
# 2) 构建反事实模型
mkdir -p /root/fedcore/cf
/root/miniconda3/envs/hpt/bin/python /root/fedcore/cf_restore.py \
  --src $K/c_int4_global_r49 --clean $K/dp_client2_r1 \
  --dst /root/fedcore/cf/c_int4_restore \
  --out /root/HPT/output/cf_c_int4_restore_report.json 2>&1 | head -30
