cd /root/HPT
rm -f /root/HPT/output/.rc_dp_s0001 /root/HPT/output/.rc_dp_s0003 /root/HPT/output/.rc_dp_s0012
nohup setsid bash /root/fedcore/chain_phase4_dp.sh > /root/HPT/output/chain_phase4_dp_rerun.log 2>&1 < /dev/null &
sleep 50
echo "=== 链进程 ==="
ps -eo pid,etime,cmd | grep -E '[c]hain_phase4_dp' | cut -c1-90
echo "=== 链日志 ==="
tail -n 14 /root/HPT/output/chain_phase4_dp_rerun.log 2>/dev/null
