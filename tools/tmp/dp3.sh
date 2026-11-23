set -u
echo "=== 时间 ==="
date '+%F %T'
echo "=== DP 三路进度 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  d=/root/HPT/output/$t
  rc=no; [ -f "/root/HPT/output/.rc_$t" ] && rc=yes
  n=$(ls "$d" 2>/dev/null | grep -c '^global_r')
  last=$(ls "$d" 2>/dev/null | grep '^global_r' | sort -V | tail -1)
  du=$(du -sh "$d" 2>/dev/null | cut -f1)
  echo "[$t] rc=$rc  n_rounds=$n  last=$last  du=$du"
done
echo "=== DP 训练日志首行（起点同构性核对）==="
for t in dp_s0001 dp_s0003 dp_s0012; do
  F="/root/HPT/output/log_${t}_train.txt"
  if [ -f "$F" ]; then
    printf '%-10s ' "$t"; head -1 "$F"
  else
    printf '%-10s ' "$t"; echo "NO log_${t}_train.txt ; ls:"; ls /root/HPT/output/ | grep -i "$t" | head -5
  fi
done
echo "=== DP 链日志（rerun）尾部 ==="
tail -n 20 /root/HPT/output/chain_phase4_dp_rerun.log 2>/dev/null
echo "=== DP 训练日志尾部 dp_s0001 ==="
tail -n 8 /root/HPT/output/log_dp_s0001_train.txt 2>/dev/null
echo "=== dplog 尾部 ==="
for t in dp_s0001 dp_s0003 dp_s0012; do echo "--- $t"; tail -n 4 "/root/HPT/output/dplog_${t}.log" 2>/dev/null; done
echo "=== 活跃训练进程 ==="
ps -eo pid,etime,cmd | grep -E '[h]pt\.run|[r]un_dp_utility|[c]hain_phase4|[p]hase4_autoshutdown' | cut -c1-110 | head -12
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1
echo "=== 磁盘 ==="
df -h /root | tail -1
