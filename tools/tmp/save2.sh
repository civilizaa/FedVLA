K=/root/fedcore/keep
for c in 1 2 3; do
  s=/root/HPT/output/c_int4/client_${c}_r49
  d=$K/c_int4_client${c}_r49
  if [ ! -d "$d" ] && [ -d "$s" ]; then cp -r "$s" "$d"; echo "backed up client$c"; fi
done
echo "=== keep 目录 ==="
du -sh $K/* 2>/dev/null
echo "=== c_bit1 进度 ==="
ls /root/HPT/output/c_bit1/ 2>/dev/null | grep '^global_r' | sort -V | tail -2
tail -n 3 /root/HPT/output/scanlog_c_bit1.log 2>/dev/null
echo "=== 磁盘 ==="
df -h /root | tail -1
