date '+%F %T'
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head
echo "=== 进程（训练/评估/守护）==="
ps -eo pid,ni,etime,rss,cmd | grep -E '[h]pt\.run|[h]pt\.run_eval|[c]hain_phase4|[p]hase4_autoshutdown|[c]f_eval_raw|[c]f_ablation' | cut -c1-105 | head -14
echo "=== load ==="
cat /proc/loadavg
echo "=== DP 轮次 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  echo "  [$t] rounds=$(ls $d 2>/dev/null | grep -c '^global_r') last=$(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -1)"
done
echo "=== DP 训练日志尾部 ==="
tail -n 4 /root/HPT/output/dplog_dp_s0001.log 2>/dev/null
echo "=== 内存 ==="
free -g | head -2
