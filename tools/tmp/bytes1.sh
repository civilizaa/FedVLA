for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  f=/root/HPT/output/$t/comm_report_r49.json
  echo "[$t] $(cat $f 2>/dev/null | tr -d '\n ' | head -c 400)"
done
echo "=== quant_drift (c_int8) ==="
cat /root/HPT/output/c_int8/quant_drift.jsonl 2>/dev/null | tail -3
