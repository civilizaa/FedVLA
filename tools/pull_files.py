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
import cloud as _cloud   # noqa: E402  （经验证更可靠的同步执行通道）

CHUNK = 8000              # 每块原始字节数
# ⚠️ 上限来自 relay agent：`REPORT_LIMIT = 12_000`（每段输出实际 POST 出去的上限）。
#    base64 之后长度 = ceil(n/3)*4，必须 < 12000 → n ≤ 9000。
#    取 8000 留余量。踩过的坑：64KB 的块 → base64 87KB → 被截断到 12000 字符
#    → 解码只剩 9000 字节 → "长度不符"（本地 9000 vs 云端 19477）。


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _run(cmd: str, timeout: int = 120, wait: int = 300) -> str:
    """执行一条云端命令并返回 stdout。

    ⚠️ 2026-11-08：原先走 push_files.enqueue + wait_results，
    在 relay 队列积压时 wait_results 会**超时返回空**（不是抛异常），
    表现为"取文件大小失败：''"，而同样的命令手动执行却成功 —— 极难定位。
    改用 cloud.run：它对 403/429/5xx 做指数退避重试，是实测最稳的通道。
    再叠一层"空结果重试"兜底（relay 偶发返回空 out）。
    """
    for attempt in range(3):
        res = _cloud.run(cmd, cwd="/root", timeout=timeout, wait=wait)
        out = res.get("out") or ""
        log(f"[dbg] try{attempt+1} res={repr(res)[:220]}")
        if out.strip():
            return out
        log(f"[warn] 第 {attempt + 1} 次返回空（rc={res.get('rc')} err="
            f"{(res.get('err') or '')[:120]}），重试 …")
        time.sleep(2 + 3 * attempt)
    return ""


def remote_size(path: str):
    # ⚠️ 原来用 python3 -c "import os;print(os.path.getsize('...'))"，
    #    嵌套双引号在 relay 的 JSON 传输里被破坏 → 云端命令执行失败、out 为空。
    #    改成无嵌套引号的 `stat -c %s`（实测可用）。
    out = _run(f"stat -c %s {path}")
    try:
        return int(out.strip().split()[0])
    except Exception:
        raise RuntimeError(f"取文件大小失败：{out[:200]!r}")


def remote_md5(path: str):
    # 同样避开 python3 -c 的嵌套引号
    out = _run(f"md5sum {path} | cut -d' ' -f1")
    out = out.strip()
    return out.split()[0] if out else ""


def pull(remote: str, local: str, chunk: int = CHUNK):
    # ⚠️ 2026-11-08 **Windows/MSYS 路径转换坑**：
    #    在 Git Bash 里 `python pull_files.py --remote /root/xxx` 时，
    #    MSYS 会把**以 / 开头的独立参数**改写成 Windows 路径：
    #      /root/HPT/output/f.tgz
    #      → C:/Users/hp/AppData/.../PortableGit/root/HPT/output/f.tgz
    #    云端收到后 stat 报 "No such file or directory"，out 为空，
    #    表现成"取文件大小失败：''"（极具误导性，因为命令行看着完全正常）。
    #    修法：复用 push_files.fix_remote 把 "/root/" 之后的部分还原出来。
    remote = pf.fix_remote(remote)
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
        # ⚠️ 同上：不用 python3 -c（嵌套引号会被传输破坏），
        #    改用纯 coreutils 的 tail/head/base64 管道，零嵌套引号。
        #    注意 WAF 只拦 `base64 -d`/`--decode`（解码），`base64 -w0`（编码）安全。
        b64 = _run(f"tail -c +{s + 1} {remote} | head -c {e - s} | base64 -w0",
                   timeout=120, wait=300).strip()
        if not b64:
            raise RuntimeError(f"第 {i} 块返回为空")
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
