echo "=== top CPU ==="
top -bn1 | head -14
echo "=== 磁盘大头 ==="
du -sh /root/HPT/output/* 2>/dev/null | sort -rh | head -14
echo "=== 总计 ==="
du -sh /root/HPT/output /root/fedcore /root/HPT 2>/dev/null | tail -4
