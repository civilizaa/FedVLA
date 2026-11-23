echo "=== DP 链进度 ==="
tail -n 6 /root/HPT/output/chain_phase4_dp.log 2>/dev/null
echo "=== 标定结果 ==="
cat /root/HPT/output/dp_calib/calib.json 2>/dev/null || echo "(无)"
echo "=== DP 目录 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  echo "[$t] $(find /root/HPT/output/$t -maxdepth 1 -type d 2>/dev/null | sort -V | tr '\n' ' ')"
done
echo "=== 失明度量 ==="
/root/miniconda3/envs/hpt/bin/python /root/fedcore/cf_blind.py \
  --quant /root/fedcore/keep/c_int4_global_r49 \
  --clean /root/fedcore/keep/c_int4_client0_r49 \
  --out /root/HPT/output/cf_blind_int4.json 2>&1 | head -30
