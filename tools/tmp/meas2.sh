PY=/root/miniconda3/envs/hpt/bin/python
rm -rf /tmp/q8
$PY /root/fedcore/cf_quant_variant.py --src /root/fedcore/keep/c_int4_client0_r49 \
   --dst /tmp/q8 --bits 8 --scope all > /dev/null 2>&1
echo "=== 客户端侧 int8 逐张量清零（自比对）==="
$PY /root/fedcore/cf_forensics.py --clean /root/fedcore/keep/c_int4_client0_r49 \
   --quant /tmp/q8 --scheme int8_g --top 14 2>&1 | sed -n '1,22p'
rm -rf /tmp/q8
