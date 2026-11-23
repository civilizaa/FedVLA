echo "--- keep dir ---"; ls -la /root/fedcore/keep/ 2>/dev/null
echo "--- r49 files ---"
for t in c_drop00 c_drop25 c_drop50 c_drop75 c_int8 c_int4 c_bit1; do
  d=/root/HPT/output/$t
  echo "[$t] $(ls $d 2>/dev/null | grep '^global_r' | sort -V | tail -3 | tr '\n' ' ')"
done
echo "--- quant_drift ---"
for t in c_int8 c_int4 c_bit1; do
  f=/root/HPT/output/$t/quant_drift.jsonl
  if [ -f "$f" ]; then echo "[$t] lines=$(wc -l < $f) tail:"; tail -n 2 "$f"; fi
done
