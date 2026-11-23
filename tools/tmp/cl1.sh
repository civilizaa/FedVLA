set -u
echo "=== 清理前 ==="
df -h /root | tail -1
rm -rf /root/HPT/output/dp_s0001 /root/HPT/output/dp_s0003 /root/HPT/output/dp_s0012
find /root/HPT/output/dp_calib -maxdepth 1 -type d -name 'global_r*' -exec rm -rf {} + 2>/dev/null
rm -f /root/HPT/output/dp_calib/opt_state_c*.pt
rm -rf /root/fedcore/cf/cf_p3c_int8b_norm /root/fedcore/cf/cf_p3c_int8b_trunk
rm -rf /root/fedcore/cf/cs_probe
echo "=== 保留的 calib ==="
ls -la /root/HPT/output/dp_calib/ 2>/dev/null
cat /root/HPT/output/dp_calib/calib.json 2>/dev/null
echo "=== keep 对照组仍完好？ ==="
ls /root/fedcore/keep/ 2>/dev/null
echo "=== 清理后 ==="
df -h /root | tail -1
