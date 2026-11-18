#!/usr/bin/env python3
"""把云端产物推送进 relay 的**结果存储**（而不是留在磁盘上等死）。

## 为什么需要它

第4阶段收工后守护脚本会调用 AutoDL 容器内的 `/usr/bin/shutdown`（kill supervisord）
来停止计费。容器一死：
  * relay agent 死亡 → `/enqueue` 派发的命令再也没人执行 → **拉文件通道断掉**；
  * 但 **relay 服务端本身跑在 WorkBuddy 托管的云主机上，不受影响**。

⇒ 只要在关机**之前**把产物写进 relay 服务端的 `/result` 存储，
   实例关机后产物依然躺在云端，本地随时能取，**不需要重开实例、不需要手动拉文件**。

## 机制

relay 服务端：
  * `POST /result`  body={id, out, err, rc} → 以 id 为键写入结果存储（保留最近 500 条，
    按 id 整数排序淘汰 ⇒ 用一个很大的 id 前缀（如 900000）就不会被后续普通命令挤掉）。
  * `GET  /results?since=N` → 返回所有 id > N 的结果。
  * 鉴权：`X-Relay-Token: <token>`（token 从 agent.py 里读，绝不打印）。

## 分块

服务端单条 `out` 上限 200_000 字符（MAX_OUT）。
文本类文件（json/md/csv）直接明文存（便于报告自动填充）；
二进制（png）按 60000 原始字节切块 → base64（编码，**不是** `base64 -d`，不触发 WAF）。

## 用法（云端）

    python3 relay_push_out.py --outdir /root/HPT/output \
        --files "comm_raw.json comm_table.md fig7_comm_drop.png"

id 基址默认 900000；本地取回时用 `GET /results?since=900000`。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "https://fedvla-relay.app.workbuddy.host"
TEXT_SUFFIX = (".json", ".md", ".csv", ".txt", ".log")
CHUNK = 60000            # 原始字节/块 → base64 后 80_000 字符 < 200_000 上限
MANIFEST_MARK = "MANIFEST:"


# ---------------------------------------------------------------- 凭据
def read_agent_creds(agent_py: str):
    """从 agent.py 里抠出 RELAY_TOKEN / RELAY_SHARECODE / RELAY_URL（不打印明文）。

    🔴 实测（2026-11-18）：**只带 X-Relay-Token 头会被网关 403**，
       必须在 URL 上追加 `sharecode=<SC>`（agent.py 本身也是这么做的）。
       带 sharecode 后 100KB 的 body 也不会被 WAF 拦。
    """
    txt = open(agent_py, encoding="utf-8", errors="ignore").read()

    def grab(name, default=""):
        m = re.search(name + r'\s*=\s*"([^"]*)"', txt)
        return m.group(1) if m else default

    return (grab("RELAY_TOKEN"), grab("RELAY_SHARECODE"),
            grab("RELAY_URL", DEFAULT_URL).rstrip("/"))


# ---------------------------------------------------------------- HTTP
def post_result(url: str, token: str, sc: str, cid: int, out: str, retries: int = 4):
    body = json.dumps({"id": cid, "out": out, "err": "", "rc": 0},
                      ensure_ascii=False).encode("utf-8")
    # 🔴 sharecode 必须挂在 URL 上，否则网关 403（token 头不够）
    full = f"{url}/result"
    full += ("&" if "?" in full else "?") + f"sharecode={sc}"
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(
                full, data=body,
                headers={"X-Relay-Token": token,
                         "Content-Type": "application/json; charset=utf-8"},
                method="POST")
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 200:
                    return True
                last = f"HTTP {r.status}"
        except Exception as e:                      # 网络抖动重试
            last = repr(e)
        print(f"    [warn] POST /result id={cid} 第{k+1}次失败：{last}", flush=True)
    return False


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="把云端产物推进 relay 结果存储")
    ap.add_argument("--agent", default="/root/HPT/agent.py")
    ap.add_argument("--outdir", default="/root/HPT/output")
    ap.add_argument("--files", required=True,
                    help='空格分隔的相对文件名（相对 --outdir）')
    ap.add_argument("--idbase", type=int, default=980000)
    ap.add_argument("--url", default="")
    args = ap.parse_args()

    token, sc, url = read_agent_creds(args.agent)
    url = (args.url or url or DEFAULT_URL).rstrip("/")
    if not token or token.startswith("__RELAY"):
        print(f"[FATAL] 没从 {args.agent} 里读到有效 RELAY_TOKEN")
        return 2
    if not sc:
        print(f"[FATAL] 没从 {args.agent} 里读到 RELAY_SHARECODE（网关必需）")
        return 2

    names = [n for n in args.files.split() if n]
    cid = args.idbase
    manifest = {"base": args.idbase, "outdir": args.outdir, "files": []}
    ok = True

    for name in names:
        path = os.path.join(args.outdir, name)
        if not os.path.isfile(path):
            print(f"[skip] {name} 不存在")
            continue
        raw = open(path, "rb").read()
        size = len(raw)
        sha = hashlib.sha256(raw).hexdigest()

        if name.lower().endswith(TEXT_SUFFIX) and size <= 150000:
            # 文本 → 明文存一份（报告自动填充直接可读）
            cid += 1
            if not post_result(url, token, sc, cid, raw.decode("utf-8", "replace")):
                ok = False
                continue
            manifest["files"].append({"name": name, "mode": "text", "id": cid,
                                      "size": size, "sha256": sha})
            print(f"[OK] {name}  text  id={cid}  {size:,}B")

        else:
            # 二进制 → 分块 base64
            n = (size + CHUNK - 1) // CHUNK if size else 0
            ids = []
            good = True
            for i in range(n):
                part = raw[i * CHUNK:(i + 1) * CHUNK]
                cid += 1
                if not post_result(url, token, sc, cid,
                                   base64.b64encode(part).decode("ascii")):
                    good = False
                    break
                ids.append(cid)
            if not good:
                ok = False
                continue
            manifest["files"].append({"name": name, "mode": "b64", "ids": ids,
                                      "size": size, "sha256": sha})
            print(f"[OK] {name}  b64   {n} 块 id={ids[0]}..{ids[-1]}  {size:,}B")

    # manifest 自己作为一条结果存进去（id 最大 ⇒ 最后淘汰）
    cid += 1
    mtxt = MANIFEST_MARK + json.dumps(manifest, ensure_ascii=False)
    if not post_result(url, token, sc, cid, mtxt):
        ok = False
    else:
        print(f"[OK] manifest  id={cid}  文件数={len(manifest['files'])}")

    print(f"[DONE] {'成功' if ok else '部分失败'}  推送到 {url}  id 范围 "
          f"{args.idbase}..{cid}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
