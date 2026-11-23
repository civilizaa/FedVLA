echo "=== 各配置目录剩余内容 ==="
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  echo "[$t] $(ls /root/HPT/output/$t 2>/dev/null | grep -E 'global_r|client_' | sort -V | tr '\n' ' ')"
done
echo "=== rc 计数 ==="
n=0; for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do [ -f /root/HPT/output/.rc_$t ] && n=$((n+1)); done
echo "comm_rc=$n/7"
echo "=== 各配置 task3 成功率 ==="
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  f=/root/HPT/output/eval_${t}_mujoco_metaworld_task3.log
  echo "[$t] $(grep '| avg' $f 2>/dev/null | tr '\n' ' ')"
done
echo "=== 进程 ==="
ps -eo pid,etime,cmd | grep -E '[e]val_model|[h]pt\.run|[r]un_comm_scan|[c]hain_phase4|[c]alibrate_dp' | head -6
