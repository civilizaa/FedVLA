echo "=== DP 三路轮次 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "[$t] global_r=$(ls $d 2>/dev/null | grep -c '^global_r') last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1) du=$(du -sh $d 2>/dev/null | cut -f1)"
done
echo "=== dp_s0001 目录构成 ==="
ls -la /root/HPT/output/dp_s0001/ 2>/dev/null | head -8
du -sh /root/HPT/output/dp_s0001/* 2>/dev/null | sort -rh | head -6
echo "=== dp_calib ==="
du -sh /root/HPT/output/dp_calib/* 2>/dev/null | sort -rh | head -5
echo "=== eval_* sweep 目录总量 ==="
du -shc /root/HPT/output/eval_* 2>/dev/null | tail -2
echo "=== 可清缓存 ==="
du -sh /root/.cache /root/miniconda3/pkgs /root/.conda 2>/dev/null
