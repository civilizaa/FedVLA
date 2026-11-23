#!/usr/bin/env python3
"""云端自举执行器：执行一条命令，把 stdout/stderr 以**高 id 段** POST 回 relay。

## 为什么必须有这个文件（2026-11-23 实测）

relay 服务端 `POST /result` 只保留**按 int(id) 排序后的末尾 500 条**：

```python
if len(_state["results"]) > 500:
    for k in sorted(_state["results"], key=lambda x: int(x))[:-500]:
        _state["results"].pop(k, None)
```

第 4 阶段用 idbase 段（980278~1000374）把存储占满后，`/enqueue` 分配到的
**普通小 id**（如 3639）结果**一写入就被淘汰** ⇒ `/poll` 里
`done = set(results.keys())` 永远不含它 ⇒ **每轮 poll 重新派发** ⇒
agent 重复执行、串行通道被堵死（实测：queued 恒为 1，命令永不返回）。

⇒ 解法：**让云端自己用高 id 段写结果**（第 4 阶段 `relay_push_out.py` 同款机制）。

## 为什么它必须自包含

"要推送才能执行、要执行才能验证推送"是个循环依赖。本文件只依赖
**标准库 + `/root/HPT/agent.py` 里的凭据**，因此它可以是**第一个**被推送上云的文件，
推送完立刻可用，循环即破。

用法（本地侧）：

    python3 /root/fedcore/relay_bootstrap.py --idbase 1005000 \
        --cmd-b64 '<base64 编码的 shell 命令>' --cwd /root/HPT
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

DEFAULT_URL = "https://fedvla-relay.app.workbuddy.host"


def read_agent_creds(path: str):
    """从 agent.py 里抠 RELAY_TOKEN / RELAY_SHARECODE / RELAY_URL（绝不打印明文）。"""
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:  # noqa: BLE001
        return "", "", DEFAULT_URL

    def grab(name: str, default: str = "") -> str:
        m = re.search(name + r'\s*=\s*"([^"]*)"', txt)
        return m.group(1) if m else default

    return (grab("RELAY_TOKEN"), grab("RELAY_SHARECODE"),
            grab("RELAY_URL", DEFAULT_URL).rstrip("/"))


def post_result(url: str, token: str, sc: str, cid: int,
                out: str, err: str, rc, retries: int = 5) -> bool:
    """把结果写进 relay 的结果存储（id 必须落在高 id 段才不会被淘汰）。"""
    body = json.dumps({"id": int(cid), "rc": rc,
                       "out": (out or "")[-190000:], "err": (err or "")[-20000:]},
                      ensure_ascii=False).encode("utf-8")
    # 🔴 sharecode 必须挂在 URL 上，否则网关 403（只带 X-Relay-Token 头不够）
    full = url + "/result"
    full += ("&" if "?" in full else "?") + "sharecode=" + sc
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(
                full, data=body, method="POST",
                headers={"X-Relay-Token": token,
                         "Content-Type": "application/json; charset=utf-8"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 200:
                    return True
                last = "HTTP %s" % r.status
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (k + 1))
    print("[FAIL] POST /result id=%s: %s" % (cid, last), flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="云端命令执行 + 高 id 段回传")
    ap.add_argument("--agent", default="/root/HPT/agent.py")
    ap.add_argument("--creds-b64", default="",
                    help="可选：base64 编码的 'token\\nsharecode\\nurl'，"
                         "用于不依赖 agent.py 位置就能回传（自举阶段用）")
    ap.add_argument("--idbase", type=int, required=True,
                    help="结果 id（必须落在高 id 段，如 1005000+）")
    ap.add_argument("--cmd-b64", default="", help="base64 编码的命令")
    ap.add_argument("--cwd", default="/root/HPT")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if not args.cmd_b64:
        print("[FATAL] 缺 --cmd-b64")
        return 2
    try:
        cmd = base64.b64decode(args.cmd_b64).decode("utf-8")
    except Exception as e:  # noqa: BLE001
        print("[FATAL] cmd-b64 解码失败:", e)
        return 2

    token, sc, url = read_agent_creds(args.agent)
    if args.creds_b64:
        try:
            parts = base64.b64decode(args.creds_b64).decode("utf-8").split("\n")
            if len(parts) >= 2 and parts[0]:
                token = parts[0]
                sc = parts[1]
                if len(parts) >= 3 and parts[2].strip():
                    url = parts[2].strip().rstrip("/")
        except Exception as e:  # noqa: BLE001
            print("[WARN] creds-b64 解析失败，沿用 agent.py:", e)
    if not token or not sc:
        print("[FATAL] 没读到 relay 凭据（token_len=%d sc_len=%d）"
              % (len(token), len(sc)))
        return 3

    try:
        p = subprocess.run(["/bin/bash", "-lc", cmd], cwd=args.cwd,
                           capture_output=True, text=True, timeout=args.timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        so = e.stdout or ""
        out = so.decode("utf-8", "replace") if isinstance(so, bytes) else so
        err = "TIMEOUT after %ss" % args.timeout
        rc = -9
    except Exception as e:  # noqa: BLE001
        out, err, rc = "", repr(e), -1

    ok = post_result(url, token, sc, args.idbase, out, err, rc)
    print("[relay_bootstrap] id=%s rc=%s posted=%s outlen=%d errlen=%d"
          % (args.idbase, rc, ok, len(out), len(err)), flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
