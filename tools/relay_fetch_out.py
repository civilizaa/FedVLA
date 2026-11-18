#!/usr/bin/env python3
"""从 relay 的结果存储里取回云端推送的产物（**实例已关机也能取**）。

## 背景

第4阶段守护脚本在收工后会自动关闭 AutoDL 容器（停止计费）。容器一死，
relay agent 也没了，`pull_files.py` 那种"下发命令→读 stdout"的通道随之失效。
但 relay **服务端**还活着，且第4阶段的产物已被守护脚本用 `relay_push_out.py`
写进了服务端的 `/result` 存储 ⇒ 本工具直接从那里取回。

## 用法

    python tools/relay_fetch_out.py --since 900000 --outdir results/cloud_final

## 原理

  * `GET /results?since=N` → 返回所有 id > N 的结果（含 `out` 字段）。
  * 结果里带 `MANIFEST:` 前缀的那条是清单：列出每个文件的
    {name, mode, id/ids, size, sha256}。
  * mode=text → out 即文件内容；mode=b64 → out 是 base64 分块，拼接后解码。
  * 每个文件都做 sha256 校验，不符即报错（**绝不写坏文件**）。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cloud as _cloud  # noqa: E402  （复用凭据加载与带重试的 HTTP 层）


def log(*a):
    print("[fetch]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description="从 relay 结果存储取回云端产物")
    ap.add_argument("--since", type=int, default=900000,
                    help="只取 id > since 的结果（与推送时的 --idbase 一致）")
    ap.add_argument("--outdir", default="results/cloud_final")
    ap.add_argument("--manifest-name", default="MANIFEST.json")
    args = ap.parse_args()

    url = _cloud.URL.rstrip("/") + "/results?since=%d" % args.since
    data = _cloud.req("/results?since=%d" % args.since, timeout=120, retries=5)
    results = data.get("results") or []
    log(f"取回 {len(results)} 条结果（id > {args.since}）")

    by_id = {}
    manifest = None
    for r in results:
        rid = int(r.get("id") or 0)
        out = r.get("out") or ""
        by_id[rid] = out
        if out.startswith("MANIFEST:"):
            manifest = json.loads(out[len("MANIFEST:"):])

    if manifest is None:
        log("[FATAL] 没找到 MANIFEST —— 守护脚本可能还没推送，或 idbase 不匹配")
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    n_ok = n_bad = 0
    for f in manifest.get("files", []):
        name = f.get("name") or ""
        dest = os.path.join(args.outdir, name)
        try:
            if f.get("mode") == "text":
                content = (by_id.get(int(f["id"])) or "").encode("utf-8")
            else:
                buf = bytearray()
                for rid in f.get("ids", []):
                    chunk = by_id.get(int(rid))
                    if not chunk:
                        raise RuntimeError(f"分块 {rid} 缺失")
                    buf += base64.b64decode(chunk)
                content = bytes(buf)
            sha = hashlib.sha256(content).hexdigest()
            if sha != f.get("sha256"):
                raise RuntimeError(f"sha256 不符：本地 {sha[:12]} vs 云端 "
                                   f"{(f.get('sha256') or '')[:12]}")
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            with open(dest, "wb") as fobj:
                fobj.write(content)
            log(f"[OK] {name}  {len(content):,}B  sha256 一致")
            n_ok += 1
        except Exception as e:
            log(f"[FAIL] {name}: {e}")
            n_bad += 1

    with open(os.path.join(args.outdir, args.manifest_name), "w",
              encoding="utf-8") as fobj:
        json.dump(manifest, fobj, ensure_ascii=False, indent=2)
    log(f"完成：成功 {n_ok}，失败 {n_bad} → {args.outdir}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
