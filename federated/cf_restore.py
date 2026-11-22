#!/usr/bin/env python3
"""反事实模型构建：把**未量化**的 normalizer 塞回量化过的全局模型。

目的
----
量化归因的核心问题：int8/int4 成功率 0.000，到底是
  (a) 权重被量化打坏了，还是
  (b) normalizer（归一化统计量）被量化清零 ⇒ 输入信息被抹掉？

做法
----
  src   = 量化后的全局模型目录（global_r49）
  clean = 客户端本地权重目录（client_i_r49）—— 量化发生在服务端聚合，
          客户端本地权重未经量化 ⇒ 其中的 normalizer 是**干净**的
  dst   = src 的副本，但所有 normalizer 张量换成 clean 里的版本

⇒ dst 与 src 的唯一差别就是 normalizer。
   在 dst 上跑评估 ⇒ 成功率若从 0 恢复，就是"normalizer 被清零"的直接因果证据；
   同时 dst 也就等价于"量化时排除 normalizer 统计量"的实现（通信量差 < 1KB）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch


def collect_clean_normalizer(clean_dir):
    """从客户端目录里收集所有 normalizer 张量：{key 路径: tensor}"""
    pool = {}
    for f in sorted(glob.glob(os.path.join(clean_dir, "*.pth")) +
                    glob.glob(os.path.join(clean_dir, "*.pt"))):
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读不了 {f}: {type(e).__name__}: {e}")
            continue
        for k, t in _walk(obj):
            if "normalizer" in k and torch.is_tensor(t) and t.is_floating_point():
                pool[k] = t.detach().float().clone()
    return pool


def _walk(node, path=""):
    """产出 (路径, 张量)"""
    if torch.is_tensor(node):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))


def _match(path, pool):
    """按路径后缀匹配 clean pool 里的张量（容忍两侧嵌套深度不同）"""
    if path in pool:
        return pool[path]
    for ck, cv in pool.items():
        if ck.endswith("." + path) or path.endswith("." + ck):
            return cv
    return None


def patch_node(node, pool, path="", stats=None):
    if stats is None:
        stats = {"replaced": 0, "missing": 0, "shape_mismatch": 0}
    if torch.is_tensor(node):
        if "normalizer" in path:
            hit = _match(path, pool)
            if hit is None:
                stats["missing"] += 1
                return node
            if hit.shape != node.shape:
                stats["shape_mismatch"] += 1
                return node
            stats["replaced"] += 1
            return hit.clone()
        return node
    if isinstance(node, dict):
        return {k: patch_node(v, pool, f"{path}.{k}" if path else str(k), stats)
                for k, v in node.items()}
    return node


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="量化后的全局模型目录（原地打补丁也允许）")
    ap.add_argument("--clean", required=True, help="客户端本地权重目录（干净 normalizer 来源）")
    ap.add_argument("--dst", default="", help="输出目录（留空＝原地修改 src）")
    ap.add_argument("--out", default="", help="报告 json 输出路径")
    args = ap.parse_args()

    dst = args.dst or args.src
    if args.dst and os.path.abspath(args.dst) != os.path.abspath(args.src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(args.src, dst)
        print(f"[i] 已复制 {args.src} → {dst}")

    pool = collect_clean_normalizer(args.clean)
    print(f"[i] 干净 normalizer 张量数 = {len(pool)}")
    for k, v in list(pool.items())[:8]:
        print(f"      {k}  shape={list(v.shape)}  absmax={float(v.abs().max()):.6g} "
              f"零数={int((v == 0).sum())}")

    report = {"src": args.src, "clean": args.clean, "dst": dst, "files": []}
    for f in sorted(glob.glob(os.path.join(dst, "*.pth")) + glob.glob(os.path.join(dst, "*.pt"))):
        fn = os.path.basename(f)
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读不了 {f}: {type(e).__name__}: {e}")
            continue
        before = {k: int((t == 0).sum()) for k, t in _walk(obj)
                  if "normalizer" in k and torch.is_tensor(t) and t.is_floating_point()}
        stats = {"replaced": 0, "missing": 0, "shape_mismatch": 0}
        new = patch_node(obj, pool, "", stats)
        after = {k: int((t == 0).sum()) for k, t in _walk(new)
                 if "normalizer" in k and torch.is_tensor(t) and t.is_floating_point()}
        if stats["replaced"]:
            torch.save(new, f)
        print(f"[i] {fn}: 替换={stats['replaced']} 缺失={stats['missing']} "
              f"形状不符={stats['shape_mismatch']}")
        if before:
            for k in list(before)[:6]:
                print(f"      {k}: 零数 {before[k]} → {after.get(k)}")
        report["files"].append({"file": fn, "stats": stats,
                                 "zeros_before": before, "zeros_after": after})

    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"[OK] 报告 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
