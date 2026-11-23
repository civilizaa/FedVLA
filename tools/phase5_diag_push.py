#!/usr/bin/env python3
"""诊断：为什么 patch_hpt_fedbaseline.py / run_fedavg.sh 推不上去。

用 relay_bootstrap 通道直接执行写入命令，捕获 stderr（普通命令的结果拿不到，
但 relay_bootstrap 会把 stdout/stderr POST 到高 id 段）。
"""
from __future__ import annotations

import base64
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402

TARGETS = ["federated/patch_hpt_fedbaseline.py", "federated/run_fedavg.sh"]
CLOUD_DIR = pr.CLOUD_DIR


def write_shell(local_path: str, remote_path: str) -> str:
    raw = open(os.path.join(ROOT, local_path), "rb").read()
    comp = gzip.compress(raw, 9)
    b64 = base64.b64encode(comp).decode("ascii")
    return (f"python3 -c \"import base64,zlib,sys;"
            f"open('{remote_path}','wb')"
            f".write(zlib.decompress(base64.b64decode(sys.argv[1]),47))\" '{b64}'")


def main() -> int:
    remote = pr.Remote()

    # ① 云端现状
    paths = " ".join(f"{CLOUD_DIR}/{os.path.basename(p)}" for p in TARGETS)
    rc, out, err = remote.run(
        f"ls -la {paths}; echo '--- md5 ---'; md5sum {paths}; "
        f"echo '--- 头部字节 ---'; head -c 120 {CLOUD_DIR}/patch_hpt_fedbaseline.py | od -c | head -4",
        timeout=180)
    print("=== [1] 云端现状 ===")
    print(out)
    if (err or "").strip():
        print("[err]", err[:600])

    # ② 本地在云端直接执行写入命令，看真实报错
    for lp in TARGETS:
        rp = f"{CLOUD_DIR}/{os.path.basename(lp)}"
        rc, out, err = remote.run(write_shell(lp, rp), timeout=300)
        print(f"\n=== [2] 直接写入 {os.path.basename(lp)} rc={rc} ===")
        print("stdout:", (out or "")[:600])
        print("stderr:", (err or "")[:1200])

    # ③ 再查 md5
    rc, out, _ = remote.run(f"md5sum {paths}", timeout=180)
    print("\n=== [3] 写入后 md5 ===")
    print(out)
    import hashlib
    for lp in TARGETS:
        want = hashlib.md5(open(os.path.join(ROOT, lp), "rb").read()).hexdigest()
        print(f"  本地 {os.path.basename(lp):<26} {want}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
