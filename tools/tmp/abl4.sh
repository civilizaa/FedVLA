cd /root/HPT
# 1) 补完 11-19 中断的 int8-trunk（复用已构建的模型，省一次量化）
nohup setsid bash -c 'TAG=cf_p3c_int8_trunk SRC=/root/fedcore/cf/cf_p3c_int8_trunk bash /root/fedcore/cf_eval.sh' > /root/HPT/output/cf_eval_trunk_retry.log 2>&1 < /dev/null &
sleep 3
# 2) 后验 int4 全量化（决定性：int4 后验是否也无损）
SCOPES="all" TAG_PREFIX=cf_p3c_int4b BITS=4 nohup setsid bash /root/fedcore/cf_ablation2.sh > /root/HPT/output/cf_ablation2_int4.log 2>&1 < /dev/null &
sleep 45
echo "=== 消融进程 ==="
ps -eo pid,etime,cmd | grep -E '[c]f_ablation2|[c]f_eval.sh|[h]pt.run_eval' | cut -c1-95 | head -8
echo "=== 消融日志 ==="
tail -n 8 /root/HPT/output/cf_ablation2.log 2>/dev/null
echo "=== DP 首轮 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "  [$t] rounds=$(ls $d 2>/dev/null | grep -c '^global_r') last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
done
tail -n 2 /root/HPT/output/autoshutdown.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
