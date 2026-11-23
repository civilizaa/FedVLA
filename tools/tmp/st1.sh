echo "=== c_int4 评估耗时（scanlog 时间戳）==="
grep -E "\[EVAL\]|\[OK\] c_int4|完成" /root/HPT/output/scanlog_c_int4.log 2>/dev/null | tail -4
echo "=== c_bit1 ==="
grep -E "\[EVAL\]|\[OK\] c_bit1" /root/HPT/output/scanlog_c_bit1.log 2>/dev/null | tail -3
echo "=== 消融结果 ==="
for t in cf_p3c_int8_all cf_p3c_int8b_norm cf_p3c_int8b_trunk; do
  F=/root/HPT/output/eval_${t}_mujoco_metaworld_task3.log
  if [ -f "$F" ]; then echo "[$t] tables=$(grep -c '| avg' $F || echo 0) $(grep '| avg' $F | tr '\n' ' ')"; fi
done
tail -n 3 /root/HPT/output/cf_ablation.log
tail -n 3 /root/HPT/output/cf_ablation2.log
echo "=== DP ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  echo "[$t] last=$(ls /root/HPT/output/$t 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
done
df -h /root | tail -1
