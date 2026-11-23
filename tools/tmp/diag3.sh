echo "--- c_int4 dir ---"; ls -la /root/HPT/output/c_int4/ 2>/dev/null | head -30
echo "--- output *.pt ---"; ls -la /root/HPT/output/*.pt 2>/dev/null | head -20
echo "--- client ckpt ---"; ls -d /root/HPT/output/*client* 2>/dev/null | head; ls -la /root/HPT/output/client0_r49 2>/dev/null | head -10
echo "--- keep ---"; ls -laR /root/fedcore/keep/ 2>/dev/null | head -20
echo "--- guard ---"; ls -la /root/HPT/output/keep_guard.log 2>/dev/null && tail -n 12 /root/HPT/output/keep_guard.log
