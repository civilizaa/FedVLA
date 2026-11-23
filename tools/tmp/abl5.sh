echo "=== 三个后验消融结果 ==="
for t in cf_p3c_int4b_all cf_p3c_int8_cs cf_p3c_int8_trunk; do
  F=/root/HPT/output/eval_${t}_mujoco_metaworld_task3.log
  if [ -f "$F" ]; then
    n=$(grep -c '| avg' "$F" 2>/dev/null || echo 0)
    echo "[$t] tables=$n $(grep '| avg' "$F" 2>/dev/null | tr '\n' ' ')"
  else echo "[$t] 无日志"; fi
done
echo "=== DP 进度 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "  [$t] rounds=$(ls $d 2>/dev/null | grep -c '^global_r') last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
done
echo "=== 守护心跳 ==="
tail -n 2 /root/HPT/output/autoshutdown.log
df -h /root | tail -1
