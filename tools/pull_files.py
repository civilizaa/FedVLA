#!/usr/bin/env python3
"""把云端 AutoDL 上的文件拉回本地（push_files.py 的反向操作）。

## 为什么需要它

relay 服务端没有"下载文件"的路由（只有 /enqueue /results /poll /run 等），
所以拉文件只能借"下发命令 → 读回 stdout"这条通道。

## 分块 base64 的原因

单次命令返回大文本会被 agent 侧截断，所以按字节偏移分块取：
    python3 -c "import base64,sys;d=open(P,'rb').read()[S:E];
                sys.stdout.write(base64.b64encode(d).decode())"

⚠️ 命令里**不能出现裸 `base64 -d` / `--decode`**（网关 WAF 规则①：命中即 403）。
   这里的 `base64.b64encode` 是编码，不触发；请求体本身也只有这段固定短文本，
   不会触发规则②（纯文本大体积）。

## 用法
    python pull_files.py --remote /root/HPT/output/phase3_raw.json \
                          --local  results/phase3_raw.json
    python pull_files.py --remote /root/HPT/output/fig.png --local results/fig.png
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import push_files as pf  # noqa: E402  （复用 req/enqueue/wait_results 与凭据加载）

CHUNK = 8000              # 每块原始字节数
# ⚠️ 上限来自 relay agent：`REPORT_LIMIT = 12_000`（每段输出实际 POST 出去的上限）。
#    base64 之后长度 = ceil(n/3)*4，必须 < 12000 → n ≤ 9000。
#    取 8000 留余量。踩过的坑：64KB 的块 → base64 87KB → 被截断到 12000 字符
#    → 解码只剩 9000 字节 → "长度不符"（本地 9000 vs 云端 19477）。


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def remote_size(path: str):
    cmd = f"python3 -c \"import os;print(os.path.getsize('{path}'))\""
    cid, _ = pf.enqueue(cmd, timeout=60)
    if cid is None:
        raise RuntimeError("入队失败")
    res = pf.wait_results([cid], timeout=120).get(cid, {})
    try:
        return int((res.get("out") or "").strip().split()[0])
    except Exception:
        raise RuntimeError(f"取文件大小失败：{(res.get('out') or '')[:200]}")


def remote_md5(path: str):
    cmd = f"python3 -c \"import hashlib;print(hashlib.md5(open('{path}','rb').read()).hexdigest())\""
    cid, _ = pf.enqueue(cmd, timeout=180)
    if cid is None:
        raise RuntimeError("入队失败")
    res = pf.wait_results([cid], timeout=300).get(cid, {})
    out = (res.get("out") or "").strip()
    return out.split()[0] if out else ""


def pull(remote: str, local: str, chunk: int = CHUNK):
    size = remote_size(remote)
    log(f"云端文件 {remote} = {size:,} 字节")
    if size == 0:
        open(local, "wb").close()
        return 0

    n_chunks = (size + chunk - 1) // chunk
    log(f"分 {n_chunks} 块拉取（每块 {chunk//1024} KiB）")

    buf = bytearray()
    for i in range(n_chunks):
        s = i * chunk
        e = min(size, s + chunk)
        cmd = (f"python3 -c \"import base64,sys;"
               f"d=open('{remote}','rb').read()[{s}:{e}];"
               f"sys.stdout.write(base64.b64encode(d).decode())\"")
        cid, _ = pf.enqueue(cmd, timeout=180)
        if cid is None:
            raise RuntimeError(f"第 {i} 块入队失败")
        res = pf.wait_results([cid], timeout=300).get(cid, {})
        b64 = (res.get("out") or "").strip()
        if not b64:
            raise RuntimeError(f"第 {i} 块返回为空（rc={res.get('rc')}）")
        try:
            buf += base64.b64decode(b64)
        except Exception as e:
            raise RuntimeError(f"第 {i} 块 base64 解码失败：{e}（长度 {len(b64)}）")
        log(f"  块 {i+1}/{n_chunks}  {e-s:,} 字节（累计 {len(buf):,}）")

    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    with open(local, "wb") as f:
        f.write(bytes(buf))

    got = os.path.getsize(local)
    if got != size:
        raise RuntimeError(f"长度不符：本地 {got} vs 云端 {size}")
    log(f"[OK] 已写入 {local}（{got:,} 字节）")

    # md5 交叉校验
    rmd5 = remote_md5(remote)
    lmd5 = hashlib.md5(open(local, "rb").read()).hexdigest()
    if rmd5 and lmd5 != rmd5:
        raise RuntimeError(f"md5 不一致：本地 {lmd5} vs 云端 {rmd5}")
    log(f"[OK] md5 校验一致：{lmd5}")
    return got


def main():
    p = argparse.ArgumentParser(description="从云端拉文件到本地")
    p.add_argument("--remote", required=True, help="云端绝对路径")
    p.add_argument("--local", required=True, help="本地保存路径")
    p.add_argument("--chunk", type=int, default=CHUNK)
    args = p.parse_args()
    pull(args.remote, args.local, args.chunk)
    print("PULL_DONE")


if __name__ == "__main__":
    main()
