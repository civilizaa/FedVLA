cd /root/HPT
SCOPES="norm trunk" TAG_PREFIX=cf_p3c_int8b nohup setsid bash /root/fedcore/cf_ablation2.sh > /root/HPT/output/cf_ablation2_launch.log 2>&1 < /dev/null &
sleep 20
ps -eo pid,etime,cmd | grep -E '[c]f_ablation' | cut -c1-70 | head -4
echo "=== DP 进度 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "[$t] last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1) evaltable=$(grep -c '| avg' /root/HPT/output/eval_${t}_mujoco_metaworld_task3.log 2>/dev/null || echo 0)"
done
