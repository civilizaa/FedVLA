cd /root/HPT/output
for f in eval_c_*.log eval_cf_*.log scanlog_c_*.log c_*/comm_report_r49.json c_*/quant_drift.jsonl dp_calib/calib.json cf_*.json; do
  [ -f "$f" ] && echo "$(stat -c%s "$f") $f"
done
echo "=== 总量 ==="
du -ch eval_c_*.log eval_cf_*.log scanlog_c_*.log c_*/comm_report_r49.json c_*/quant_drift.jsonl dp_calib/calib.json cf_*.json 2>/dev/null | tail -1
