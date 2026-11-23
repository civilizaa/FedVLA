echo "=== p3_moe_eda_cluster 评估日志 ==="
ls /root/HPT/output/eval_p3_moe_eda_cluster_mujoco_metaworld_task*/ 2>/dev/null | head
for d in 1 2 3; do
  f=/root/HPT/output/eval_p3_moe_eda_cluster_mujoco_metaworld_task$d.log
  [ -f "$f" ] && echo "task$d: $(grep '| avg' $f | tr '\n' ' ')"
done
echo "=== 模型完整性 ==="
ls -la /root/HPT/output/p3_moe_eda_cluster/global_r49/ | head -8
echo "=== 反事实评估 ==="
F=/root/HPT/output/eval_cf_int4_restore_mujoco_metaworld_task3.log
echo "tables=$(grep -c '| avg' $F 2>/dev/null || echo 0)"
grep '| avg' $F 2>/dev/null | tr '\n' ' '
