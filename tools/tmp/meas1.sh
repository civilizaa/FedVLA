PY=/root/miniconda3/envs/hpt/bin/python
echo "=== 客户端侧真实清零率（自比对，无聚合污染）==="
for b in 8 4; do
  echo "--- bits=$b ---"
  $PY /root/fedcore/cf_quant_variant.py --src /root/fedcore/keep/c_int4_client0_r49 \
     --dst /tmp/qtest_$b --bits $b --scope all 2>&1 | grep -E "量化张量数|→"
  rm -rf /tmp/qtest_$b
done
echo "=== 消融进展 ==="
tail -n 4 /root/HPT/output/cf_ablation.log
for t in cf_p3c_int8_all cf_p3c_int8_norm cf_p3c_int8_trunk; do
  F=/root/HPT/output/eval_${t}_mujoco_metaworld_task3.log
  [ -f "$F" ] && echo "[$t] tables=$(grep -c '| avg' $F || echo 0)"
done
