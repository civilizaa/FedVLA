date '+%F %T'
echo "--- load ---"; cat /proc/loadavg; nproc
echo "--- dirs ---"; ls -d /root/HPT/output/*/ 2>/dev/null | head -20
echo "--- rounds ---"
for t in c_int4 c_bit1 c_int8; do
  d=/root/HPT/output/$t
  if [ -d "$d" ]; then
    echo "[$t] r_files=$(ls "$d" 2>/dev/null | grep -c '^global_r') last=$(ls "$d" 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
  else
    echo "[$t] no dir $d"
  fi
done
