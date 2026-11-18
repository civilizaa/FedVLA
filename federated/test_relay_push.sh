#!/usr/bin/env bash
# relay 产物推送通道的往返测试（云端侧）
#   1) 造一个文本文件 + 一个 130KB 随机二进制文件
#   2) 调 relay_push_out.py 推进 relay 结果存储（测试用 idbase=950000）
#   3) 打印两个文件的 md5，供本地比对
set -u
OUT=/root/HPT/output
PY=/root/miniconda3/envs/hpt/bin/python
mkdir -p "$OUT"

printf 'hello relay\nsecond line\n' > "$OUT/relay_push_test.txt"
head -c 130000 /dev/urandom > "$OUT/relay_push_test.bin"

"$PY" /root/fedcore/relay_push_out.py \
    --outdir "$OUT" \
    --files "relay_push_test.txt relay_push_test.bin" \
    --idbase 950000
echo "PUSH_RC=$?"

echo "--- 云端 md5 ---"
md5sum "$OUT/relay_push_test.txt" "$OUT/relay_push_test.bin"
echo "TEST_PUSH_DONE"
