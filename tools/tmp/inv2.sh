echo "=== 剩余权重目录（只看目录）==="
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  d=$(find /root/HPT/output/$t -maxdepth 1 -type d \( -name 'global_r*' -o -name 'client_*_r*' \) 2>/dev/null | sort -V | tr '\n' ' ')
  echo "[$t] $d"
done
echo "=== rc ==="
n=0; for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do [ -f /root/HPT/output/.rc_$t ] && n=$((n+1)); done
echo "comm_rc=$n/7"
echo "=== task3 成功率 ==="
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  echo "[$t] $(grep '| avg' /root/HPT/output/eval_${t}_mujoco_metaworld_task3.log 2>/dev/null | tr '\n' ' ')"
done
echo "=== 进程 ==="
ps -eo pid,etime,cmd | grep -E '[e]val_model|[h]pt\.run|[r]un_comm_scan|[c]hain_phase4|[c]alibrate_dp' | cut -c1-120 | head -6
