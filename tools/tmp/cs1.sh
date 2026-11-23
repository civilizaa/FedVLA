# 1) 停掉重复的顺序版消融（并行版已覆盖 norm/trunk）
pkill -f '[c]f_ablation.sh' 2>/dev/null || true
sleep 2
pkill -f '[e]val_cf_p3c_int8_norm' 2>/dev/null || true
ps -eo pid,cmd | grep -E '[c]f_ablation' | cut -c1-60
# 2) 构建"客户端步长"变体并测量（不评估，只看损伤水平）
PY=/root/miniconda3/envs/hpt/bin/python
$PY /root/fedcore/cf_quant_variant.py \
  --src /root/fedcore/keep/p3c_global_r49 \
  --dst /root/fedcore/cf/cs_probe \
  --bits 8 --scope all \
  --absmax-from /root/fedcore/keep/c_int4_client0_r49 2>&1 | grep -E "步长参照|量化张量数"
rm -rf /root/fedcore/cf/cs_probe
