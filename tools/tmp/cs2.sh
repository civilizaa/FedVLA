PY=/root/miniconda3/envs/hpt/bin/python
echo "=== cf_quant_variant.py 是否支持 --absmax-from ==="
$PY /root/fedcore/cf_quant_variant.py --help 2>&1 | head -20
