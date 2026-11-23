echo "=== 恢复后模型的失明度量（task3 应恢复到 39/39）==="
/root/miniconda3/envs/hpt/bin/python /root/fedcore/cf_blind.py \
  --quant /root/fedcore/cf/cf_int4_restore \
  --out /root/HPT/output/cf_blind_int4restore.json 2>&1 | head -12
echo "=== 评估耗时对比 ==="
for f in eval_c_int4_mujoco_metaworld_task3 eval_cf_int4_restore_mujoco_metaworld_task3; do
  L=/root/HPT/output/$f.log
  echo "[$f] $(grep -c '| avg' $L 2>/dev/null) 表  首行时间=$(grep -oE '^[0-9-]+ [0-9:]+' $L 2>/dev/null | head -1)"
done
echo "=== 消融进展 ==="
tail -n 6 /root/HPT/output/cf_ablation.log
