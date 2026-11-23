cd /root/HPT/output
FILES=""
for f in eval_c_*.log eval_cf_*.log scanlog_c_*.log c_*/comm_report_r49.json c_*/quant_drift.jsonl dp_calib/calib.json cf_*.json; do
  [ -f "$f" ] && FILES="$FILES $f"
done
echo "文件数: $(echo $FILES | wc -w)"
nohup setsid python3 /root/fedcore/relay_push_out.py --outdir /root/HPT/output --files "$FILES" --idbase 990000 > /root/HPT/output/relay_push_manual.log 2>&1 < /dev/null &
sleep 30
echo "=== 推送日志 ==="
tail -n 12 /root/HPT/output/relay_push_manual.log 2>/dev/null
