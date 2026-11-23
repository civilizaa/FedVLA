for t in cf_int4_restore cf_p3c_int8_all cf_p3c_int8_norm cf_p3c_int8_trunk cf_p3c_int8b_norm cf_p3c_int8b_trunk; do
  F=/root/HPT/output/eval_${t}_mujoco_metaworld_task3.log
  if [ -f "$F" ]; then
    n=$(grep -c '| avg' "$F" 2>/dev/null || echo 0)
    echo "[$t] tables=$n $(grep '| avg' "$F" 2>/dev/null | tr '\n' ' ')"
  else
    echo "[$t] 无日志"
  fi
done
echo "=== 消融日志尾部 ==="
tail -n 10 /root/HPT/output/cf_ablation.log 2>/dev/null
echo "--- ab2 ---"
tail -n 10 /root/HPT/output/cf_ablation2.log 2>/dev/null
