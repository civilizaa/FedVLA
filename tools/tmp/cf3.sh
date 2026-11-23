cd /root/HPT
TAG=cf_int4_restore \
SRC=/root/fedcore/keep/c_int4_global_r49 \
CLEAN=/root/fedcore/keep/dp_client2_r1 \
nohup setsid bash /root/fedcore/cf_eval.sh > /root/HPT/output/cf_eval_launch.log 2>&1 < /dev/null &
sleep 45
echo "=== 启动日志 ==="
head -25 /root/HPT/output/cf_eval_launch.log 2>/dev/null
echo "=== 进程 ==="
ps -eo pid,etime,cmd | grep -E '[c]f_eval.sh|[h]pt.run_eval' | cut -c1-90 | head -4
