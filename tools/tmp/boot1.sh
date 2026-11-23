set -u
echo "=== 时间/负载 ==="
date '+%F %T'; cat /proc/loadavg; nproc; df -h /root | tail -1
echo "=== 进程（真实存活）==="
ps -eo pid,etime,cmd | grep -E '[h]pt\.run|[e]val_model|[r]un_comm_scan|[c]hain_phase4|[c]f_ablation|[r]un_dp|[c]alibrate_dp' | cut -c1-110 | head -10
echo "=== output 顶层 ==="
ls -d /root/HPT/output/*/ 2>/dev/null | head -30
echo "=== 通信扫描 7 配置 rc ==="
n=0; for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  [ -f /root/HPT/output/.rc_$t ] && n=$((n+1))
done; echo "comm_rc=$n/7"
echo "=== DP 三路 rc ==="
d=0; for t in dp_s0001 dp_s0003 dp_s0012; do
  [ -f /root/HPT/output/.rc_$t ] && d=$((d+1))
done; echo "dp_rc=$d/3"
echo "=== keep 备份（反事实对照组）==="
ls -la /root/fedcore/keep/ 2>/dev/null
echo "=== DP 结果文件 ==="
ls -la /root/HPT/output/dp_utility_raw.json /root/HPT/output/dp_calib/calib.json 2>/dev/null
echo "=== 关机/守护日志尾部 ==="
tail -n 12 /root/HPT/output/autoshutdown.log 2>/dev/null
echo "=== 链日志尾部 ==="
tail -n 10 /root/HPT/output/chain_phase4_dp.log 2>/dev/null
