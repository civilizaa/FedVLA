cd /root/HPT
export WAIT_DP=1 MAX_HOURS=8 POLL=60s PUSH_IDBASE=1000000
nohup setsid bash /root/fedcore/phase4_autoshutdown.sh > /root/HPT/output/autoshutdown.log 2>&1 < /dev/null &
sleep 40
echo "=== 守护进程 ==="
ps -eo pid,etime,cmd | grep -E '[p]hase4_autoshutdown' | cut -c1-90
echo "=== 守护日志 ==="
tail -n 8 /root/HPT/output/autoshutdown.log 2>/dev/null
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
echo "=== 训练进程 ==="
ps -eo pid,etime,cmd | grep -E '[h]pt\.run' | cut -c1-100 | head -5
