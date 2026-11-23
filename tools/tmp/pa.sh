echo "=== fedcore 内容 ==="
ls -la /root/fedcore/ 2>/dev/null | head -30
echo "=== relay_push_out.py 用法 ==="
head -30 /root/fedcore/relay_push_out.py 2>/dev/null || echo "不存在"
echo "=== HF cache（不能删）==="
du -sh /root/.cache/* 2>/dev/null | sort -rh | head -6
echo "=== cf 目录 ==="
du -sh /root/fedcore/cf/* 2>/dev/null
