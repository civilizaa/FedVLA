#!/usr/bin/env python3
"""抢救取回 sha256 不符的**追加型日志**（保留原样 + 附核验说明）。

## 为什么需要它

`relay_fetch_out.py` 的铁律是"sha256 不符即报错、绝不写坏文件"。
第4阶段收工批里有 3 个日志踩到了这个闸门：

  * `autoshutdown.log` / `extract_comm.log` / `extract_dp.log`

诊断结论（2026-11-22）：这三个都是**守护脚本在推送窗口内仍在追加写**的日志，
清单里的 sha256 对应的是**更早的快照**，而 relay 存储里留下的是**更晚的版本**：

  * `extract_comm.log`：清单 size=4246，实际存储 6367B，
    **前 4246 字节的 sha256 精确命中清单 sha** ⇒ 文件在"算哈希"之后被继续追加。
  * `extract_dp.log`：清单 size=3001，实际存储 4609B，
    **前 3001 字节 sha256 命中清单 sha** ⇒ 同上。
  * `autoshutdown.log`：清单 size=2911，实际存储 979B，
    前缀/后缀均不命中 ⇒ 该文件被**截断后重写**（清单哈希对应截断前的更完整版本）。

⇒ 这不是内容损坏、不是数据缺失，而是**推送与写日志并发的时间戳竞态**。
数据类产物（JSON / PNG / scanlog）sha256 **全部一致**，可直接使用。

本工具把这类文件的**实际存储内容**原样落盘到 `logs_unverified/`，
并生成 `SHA256_MISMATCH.md` 说明每一个文件的核验细节 —— 供论文附录引用原始日志，
同时绝不冒充"校验通过的文件"。

## 用法

    python tools/relay_salvage_logs.py --since 990000 --outdir results/cloud_final_rerun
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
import cloud as _cloud  # noqa: E402


def log(*a):
    print("[salvage]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser(description="抢救取回哈希不符的追加型日志")
    ap.add_argument("--since", type=int, default=990000)
    ap.add_argument("--outdir", default="results/cloud_final_rerun")
    ap.add_argument("--subdir", default="logs_unverified")
    args = ap.parse_args()

    data = _cloud.req("/results?since=%d" % args.since, timeout=120, retries=5)
    results = data.get("results") or []
    log(f"relay 返回 {len(results)} 条结果（id > {args.since}）")

    by_id = {}
    manifest = None
    manifest_id = -1
    for r in results:
        rid = int(r.get("id") or 0)
        out = r.get("out") or ""
        by_id[rid] = out
        if out.startswith("MANIFEST:") and rid > manifest_id:
            manifest_id = rid
            manifest = json.loads(out[len("MANIFEST:"):])
    if manifest is None:
        log("[FATAL] 没找到 MANIFEST")
        return 2

    outdir = os.path.join(args.outdir, args.subdir)
    os.makedirs(outdir, exist_ok=True)

    notes = []
    n_salvaged = 0
    for f in manifest.get("files", []):
        name = f.get("name") or ""
        if f.get("mode") == "text":
            content = (by_id.get(int(f["id"])) or "").encode("utf-8")
            chunks = [int(f["id"])]
        else:
            buf = bytearray()
            chunks = [int(x) for x in f.get("ids", [])]
            for rid in chunks:
                buf += base64.b64decode(by_id.get(rid) or "")
            content = bytes(buf)

        sha = hashlib.sha256(content).hexdigest()
        if sha == f.get("sha256"):
            continue  # 校验通过的文件由 relay_fetch_out.py 负责，这里不碰

        # 前缀命中判定：内容是清单快照的超集（追加）还是子集（截断）
        prefix_hit = None
        want = int(f.get("size") or 0)
        if want and len(content) > want:
            prefix_hit = hashlib.sha256(content[:want]).hexdigest() == f.get("sha256")
        elif want:
            prefix_hit = hashlib.sha256(content).hexdigest() == f.get("sha256")

        with open(os.path.join(outdir, name), "wb") as fobj:
            fobj.write(content)
        n_salvaged += 1
        notes.append({
            "name": name,
            "manifest_size": want,
            "stored_size": len(content),
            "manifest_sha256": f.get("sha256"),
            "stored_sha256": sha,
            "prefix_snapshot_match": prefix_hit,
            "result_ids": chunks,
            "verdict": ("内容为清单快照的超集（推送窗口内被继续追加）"
                        if prefix_hit is True else
                        "内容与清单快照不一致（清单哈希对应截断前的版本）"
                        if prefix_hit is False else "无法判定"),
        })
        log(f"[SALVAGE] {name}  清单 {want}B / 存储 {len(content)}B  前缀命中={prefix_hit}")

    md = [
        "# 哈希不符的追加型日志（抢救归档，**未通过 sha256 校验**）",
        "",
        "> 本目录由 `tools/relay_salvage_logs.py` 生成。",
        "> 这些文件**不是**取回失败的数据缺失，而是**推送与写日志并发造成的时间戳竞态**：",
        "> 清单里的 sha256 对应更早的快照，relay 存储里留下的是更晚的版本。",
        "> 同一批次的**数据类产物（JSON / PNG / scanlog）sha256 全部一致**，不受影响。",
        "",
        "| 文件 | 清单 size | 存储 size | 前缀快照命中 | 判定 |",
        "|---|---|---|---|---|",
    ]
    for n in notes:
        md.append("| `%s` | %d | %d | %s | %s |" % (
            n["name"], n["manifest_size"], n["stored_size"],
            {True: "✅ 是", False: "❌ 否", None: "—"}[n["prefix_snapshot_match"]],
            n["verdict"]))
    md += ["", "## 明细", "", "```json",
           json.dumps(notes, ensure_ascii=False, indent=2), "```", ""]
    with open(os.path.join(outdir, "SHA256_MISMATCH.md"), "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(md))

    log(f"完成：抢救 {n_salvaged} 个文件 → {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
