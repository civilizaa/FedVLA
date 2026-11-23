#!/usr/bin/env python3
"""通用云端命令执行 / 文件取回（走 relay_bootstrap 高 id 段通道）。

用法:
    python tools/cloud_exec.py "shell 命令"              # 执行并打印
    python tools/cloud_exec.py --get /root/fedcore/x.sh   # 取回文件到本地
    python tools/cloud_exec.py --out 文件 "shell 命令"     # 执行并保存输出
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import phase5_remote as pr  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    get_path = None
    out_path = None
    cmd = None
    i = 0
    while i < len(args):
        if args[i] == "--get":
            i += 1
            get_path = args[i]
        elif args[i] == "--out":
            i += 1
            out_path = args[i]
        else:
            cmd = args[i]
        i += 1

    remote = pr.Remote()
    if get_path:
        cmd = f"cat {get_path}"
    if not cmd:
        print(__doc__)
        return 2
    rc, out, err = remote.run(cmd, timeout=300)
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print(f"[OK] 已保存 {len(out):,}B → {out_path}（rc={rc}）")
        if err.strip():
            print("[err]", err[:800])
        return 0 if rc == 0 else 1
    print(f"--- rc={rc} ---")
    print(out)
    if (err or "").strip():
        print("--- err ---")
        print(err[:2000])
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
