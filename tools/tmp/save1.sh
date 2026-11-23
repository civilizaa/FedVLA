set -u
K=/root/fedcore/keep
mkdir -p $K
# 只拷 c_int4 的 r49（源目录此刻可能仍在写入，先看稳定性）
echo "--- 源目录状态 ---"
ls -la /root/HPT/output/c_int4/global_r49/ 2>/dev/null | head -8
sleep 8
echo "--- 8 秒后再看 ---"
ls -la /root/HPT/output/c_int4/global_r49/ 2>/dev/null | head -8
if [ ! -d "$K/c_int4_global_r49" ]; then cp -r /root/HPT/output/c_int4/global_r49 $K/c_int4_global_r49; fi
if [ ! -d "$K/c_int4_client0_r49" ]; then cp -r /root/HPT/output/c_int4/client_0_r49 $K/c_int4_client0_r49; fi
echo "--- 备份结果 ---"
du -sh $K/c_int4_global_r49 $K/c_int4_client0_r49 2>/dev/null
ls -la $K/c_int4_global_r49/ $K/c_int4_client0_r49/ 2>/dev/null | head -20
df -h /root | tail -1
