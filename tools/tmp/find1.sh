echo "=== output 顶层目录 ==="
ls -d /root/HPT/output/*/ 2>/dev/null | head -40
echo "=== 含 model.pth 的目录（非 eval_/dp_/c_ 前缀）==="
find /root/HPT/output -maxdepth 3 -name model.pth 2>/dev/null | grep -vE '/(eval_|dp_s|c_drop|c_int|c_bit)' | head -20
echo "=== 反事实评估状态 ==="
tail -8 tools/cf_poll.log 2>/dev/null
F=/root/HPT/output/eval_cf_int4_restore_mujoco_metaworld_task3.log
echo "tables=$(grep -c '| avg' $F 2>/dev/null || echo 0)"
grep -oE "number of params \(M\): [0-9.]+" $F 2>/dev/null | head -2
