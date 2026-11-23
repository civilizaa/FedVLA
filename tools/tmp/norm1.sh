PY=/root/miniconda3/envs/hpt/bin/python
for b in 8 4; do
  rm -rf /tmp/nq$b
  $PY /root/fedcore/cf_quant_variant.py --src /root/fedcore/keep/dp_client2_r1 \
     --dst /tmp/nq$b --bits $b --scope all > /dev/null 2>&1
  echo "=== task3 干净 normalizer 被 int$b 量化的结果 ==="
  $PY /root/fedcore/cf_blind.py --quant /tmp/nq$b 2>&1 | grep -E "task3|state|action" | head -4
  rm -rf /tmp/nq$b
done
echo "=== 干净基线 ==="
$PY /root/fedcore/cf_blind.py --quant /root/fedcore/keep/dp_client2_r1 2>&1 | grep task3 | head -3
