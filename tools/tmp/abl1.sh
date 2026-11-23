cd /root/HPT
nohup setsid bash /root/fedcore/cf_ablation.sh > /root/HPT/output/cf_ablation_launch.log 2>&1 < /dev/null &
sleep 25
echo "=== 进程 ==="
ps -eo pid,etime,cmd | grep -E '[c]f_ablation.sh' | cut -c1-80 | head -3
echo "=== 消融日志 ==="
tail -n 12 /root/HPT/output/cf_ablation.log 2>/dev/null
echo "=== 反事实评估 ==="
F=/root/HPT/output/eval_cf_int4_restore_mujoco_metaworld_task3.log
echo "tables=$(grep -c '| avg' $F 2>/dev/null || echo 0)  $(grep '| avg' $F 2>/dev/null | tr '\n' ' ')"
