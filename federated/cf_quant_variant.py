#!/usr/bin/env python3
"""后验量化变体：对**已训练好**的干净模型按不同范围做 per-group 量化。

用途
----
第4阶段的量化是"训练过程中逐轮量化上行"。为了把"到底是哪部分被量化打坏"隔离干净，
这里做**后验消融**：拿一个干净模型（DP 阶段产出的全局模型，量化/DP 噪声之外），
分别只对不同范围做量化，然后评估：

  scope=all   → 与链路一致（所有共享参数：trunk / normalizer / modalities_tokens）
  scope=norm  → **只**量化 normalizer（trunk 保持干净）
  scope=trunk → **只**量化 trunk 与 modalities_tokens（normalizer 保持干净）

⇒ 若 norm 变体崩溃而 trunk 变体正常 ⇒ 致命环节是 normalizer；
⇒ 若 trunk 变体也崩溃 ⇒ 权重损伤本身就足以摧毁策略。

⚠️ 与链路的**口径一致性**：量化实现直接复用 `comm_analysis.quantize_per_group`，
   参数筛选复用 `comm_channel._is_shared_param`（stems.*/heads.* 不上行、不量化）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, "/root/fedcore")
from comm_analysis import quantize_per_group  # noqa: E402


def is_shared(k: str) -> bool:
    """与 comm_channel._is_shared_param 同口径：个性化参数（stems/heads）不上行。"""
    return not (k.startswith("stems.") or k.startswith("heads."))


def in_normalizer(k: str) -> bool:
    return "normalizer" in k


def quantize_with_ref(t: torch.Tensor, bits: int, ref: torch.Tensor,
                      group_size: int = 64) -> torch.Tensor:
    """按**参照模型**的 per-group absmax 定步长，量化目标张量 t。

    用途：复现"训练中逐轮量化"的真实破坏程度。
      客户端本地权重的 per-group absmax 明显大于聚合后全局模型的 ⇒ 步长更大 ⇒
      清零更多（实测 7.69% vs 0.46%）。本函数让"对聚合模型做量化"也能用客户端的步长，
      从而在**不重训**的前提下把 in-training 的损伤水平搬到干净基线上做评估。
    """
    qmax = 2 ** (bits - 1) - 1
    tf = t.detach().float().reshape(-1)
    rf = ref.detach().float().reshape(-1)
    n = tf.numel()
    out = torch.empty_like(tf)
    for start in range(0, n, group_size):
        end = min(start + group_size, n)
        row = tf[start:end]
        s = float(rf[start:end].abs().max()) / qmax
        if s == 0:
            out[start:end] = row
        else:
            q = torch.clamp(torch.round(row / s), -qmax - 1, qmax)
            out[start:end] = q * s
    return out.reshape(t.shape).to(t.dtype)


def quantize_obj(obj, scheme_bits, scope, stats, ref=None):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        return {k: _leaf(v, k, scheme_bits, scope, stats, ref) for k, v in obj.items()}
    return obj


def _leaf(v, key, bits, scope, stats, ref=None):
    if not (torch.is_tensor(v) and v.is_floating_point()):
        return v
    want = False
    if scope == "all":
        want = is_shared(key)
    elif scope == "norm":
        want = in_normalizer(key)
    elif scope == "trunk":
        want = is_shared(key) and not in_normalizer(key)
    if not want:
        return v
    if ref is not None and key in ref and ref[key].shape == v.shape:
        q = quantize_with_ref(v, bits, ref[key])
    else:
        q = quantize_per_group(v.detach().float(), bits, 64).to(v.dtype)
    nz = int((v != 0).sum())
    killed = int(((v != 0) & (q == 0)).sum())
    rel = float(torch.linalg.norm(v.float() - q.float()) / (torch.linalg.norm(v.float()) + 1e-12))
    stats["tensors"] += 1
    stats["killed"] += killed
    stats["nonzero"] += nz
    stats["rel_sum"] += rel
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--scope", choices=["all", "norm", "trunk"], default="all")
    ap.add_argument("--absmax-from", default="",
                    help="参照模型目录：按它的 per-group absmax 定步长（复现 in-training 损伤水平）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if os.path.isdir(args.dst):
        shutil.rmtree(args.dst)
    shutil.copytree(args.src, args.dst)
    print(f"[i] {args.src} → {args.dst}（scope={args.scope} bits={args.bits}）")

    # 参照模型的逐张量张量（用于按它的 absmax 定步长）
    ref_map = {}
    if args.absmax_from:
        for f in sorted(glob.glob(os.path.join(args.absmax_from, "*.pth"))):
            try:
                obj = torch.load(f, map_location="cpu", weights_only=False)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if torch.is_tensor(v) and v.is_floating_point():
                        ref_map[k] = v.detach().float()
        print(f"[i] 步长参照模型 = {args.absmax_from}（{len(ref_map)} 个张量）")

    stats = {"tensors": 0, "killed": 0, "nonzero": 0, "rel_sum": 0.0}
    for f in sorted(glob.glob(os.path.join(args.dst, "*.pth"))):
        fn = os.path.basename(f)
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {fn}: {type(e).__name__}")
            continue
        new = quantize_obj(obj, args.bits, args.scope, stats, ref_map)
        if stats["tensors"]:
            torch.save(new, f)
            print(f"[i] {fn}: 量化张量数={stats['tensors']} "
                  f"清零={stats['killed']}/{stats['nonzero']} "
                  f"({100.0 * stats['killed'] / max(stats['nonzero'], 1):.3f}%) "
                  f"mean_rel={stats['rel_sum'] / max(stats['tensors'], 1):.5f}")
            stats = {"tensors": 0, "killed": 0, "nonzero": 0, "rel_sum": 0.0}

    rep = {"src": args.src, "dst": args.dst, "scope": args.scope, "bits": args.bits}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
