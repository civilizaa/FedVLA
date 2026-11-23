set -u
PY=/root/miniconda3/envs/hpt/bin/python
cd /root/HPT
echo "=== 构建：P3-C 全局权重 + 客户端式步长（int8）==="
rm -rf /root/fedcore/cf/cf_p3c_int8_cs
$PY /root/fedcore/cf_quant_variant.py --src /root/fedcore/keep/p3c_global_r49 \
   --dst /root/fedcore/cf/cf_p3c_int8_cs --bits 8 --scope all \
   --absmax-from /root/fedcore/keep/c_int4_client0_r49 2>&1 | tail -8
echo "=== 起评估 ==="
nohup setsid bash -c 'TAG=cf_p3c_int8_cs SRC=/root/fedcore/cf/cf_p3c_int8_cs bash /root/fedcore/cf_eval.sh' > /root/HPT/output/cf_eval_cs.log 2>&1 < /dev/null &
sleep 35
ps -eo pid,etime,cmd | grep -E '[c]f_eval.sh|[h]pt.run_eval' | cut -c1-85 | head -6
