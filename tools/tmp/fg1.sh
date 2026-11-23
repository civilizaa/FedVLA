echo "=== 旧守望进程 ==="
ps -eo pid,etime,cmd | grep -E '[k]eep_guard' || echo "无"
pkill -f '[k]eep_guard' 2>/dev/null || true
sleep 1
echo "=== 起新守望（c_bit1）==="
TAGS="c_bit1" nohup setsid bash /root/fedcore/cf_keep.sh > /dev/null 2>&1 < /dev/null &
sleep 3
ps -eo pid,etime,cmd | grep -E '[c]f_keep.sh' || echo "守望未起"
echo "=== 当前评估/训练进程 ==="
ps -eo pid,etime,cmd | grep -E '[e]val_model|[h]pt\.run|[r]un_fedavg|[r]un_comm_scan' | head -8
echo "=== 取证 int4 ==="
cd /root/fedcore && /root/miniconda3/envs/hpt/bin/python /root/fedcore/cf_forensics.py \
  --clean /root/fedcore/keep/c_int4_client0_r49 \
  --quant /root/fedcore/keep/c_int4_global_r49 \
  --scheme int4_g --out /root/HPT/output/cf_forensics_int4.json 2>&1 | head -70
