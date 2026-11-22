#!/usr/bin/env python3
"""量化 ⇒ 状态信息维度坍缩 的数值度量（纯 CPU，秒级，不跑 MuJoCo）。

要证明的事
----------
HPT 的归一化是  y = x·scale + offset  （normalizer.py 的 _normalize, forward=True）。
若量化把 scale 的某些维度清零 ⇒ 那些维度上 **y ≡ offset（常数）**
⇒ 原始状态 x 的信息在该维度被**彻底抹掉**
⇒ 策略的输入里这��维不再携带任何信息 ⇒ 策略对状态"失明"。

本脚本给三个数字
--------------
  1. dead_dims：scale 中被清零的维度数（= 信息丢失的维度数）
  2. 归一化状态 z 的逐维方差（用随机 x 抽样）：dead dim 方差恒为 0
  3. 有效状态维度数 = 方差 > 0 的维度数  →  "39 维状态只剩 k 维有效"

对照：--clean 给一份**未量化**的模型目录（客户端本地权重），
      同一套度量在干净权重上跑一遍 ⇒ 干净模型 39 维全有效。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch


def load_sd(d):
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.pth")) + glob.glob(os.path.join(d, "*.pt"))):
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    sd[k] = v.detach().float()
    return sd


def domains_of(sd):
    doms = set()
    for k in sd:
        if k.startswith("normalizer.") and k.count(".") >= 2:
            doms.add(k.split(".")[1])
    return sorted(doms)


def measure(sd, dom, n_samples=256, seed=0):
    """返回该域的状态/动作归一化器度量"""
    out = {"domain": dom}
    g = torch.Generator().manual_seed(seed)
    for mod in ("state", "action"):
        s_key = f"normalizer.{dom}.params_dict.{mod}.scale"
        o_key = f"normalizer.{dom}.params_dict.{mod}.offset"
        if s_key not in sd:
            continue
        scale = sd[s_key].reshape(-1)
        offset = sd[o_key].reshape(-1) if o_key in sd else torch.zeros_like(scale)
        n = int(scale.numel())
        dead = (scale == 0)
        dead_n = int(dead.sum())
        # 归一化输出的逐维方差：y = x*scale + offset，Var(y) = Var(x)*scale^2
        x = torch.randn(n_samples, n, generator=g)
        y = x * scale.unsqueeze(0) + offset.unsqueeze(0)
        var = y.var(dim=0)
        eff = int((var > 1e-12).sum())
        out[mod] = {
            "dim": n,
            "zero_scale_dims": dead_n,
            "effective_dims": eff,
            "scale_absmax": float(scale.abs().max()),
            "scale_min_nonzero": float(scale[scale != 0].abs().min()) if int((scale != 0).sum()) else 0.0,
            "offset_absmax": float(offset.abs().max()),
            "var_max": float(var.max()),
            "var_mean_live": float(var[var > 1e-12].mean()) if eff else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", required=True, help="量化后的模型目录")
    ap.add_argument("--clean", default="", help="未量化的对照模型目录（可选）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rep = {"quant_dir": args.quant, "clean_dir": args.clean}
    qsd = load_sd(args.quant)
    rep["quant"] = [measure(qsd, d) for d in domains_of(qsd)]

    if args.clean:
        csd = load_sd(args.clean)
        rep["clean"] = [measure(csd, d) for d in domains_of(csd)]
        # 同一域的对照（量化模型里没有的域，clean 有也算）
        rep["clean_only_domains"] = sorted(set(domains_of(csd)) - set(domains_of(qsd)))

    print("=== 量化模型（检查点里就是被量化损伤的那份）===")
    for r in rep["quant"]:
        for mod in ("state", "action"):
            if mod not in r:
                continue
            m = r[mod]
            print(f"  [{r['domain']:<24}] {mod:<6} dim={m['dim']:<3} "
                  f"scale被清零={m['zero_scale_dims']:<3} 有效维度={m['effective_dims']:<3} "
                  f"absmax={m['scale_absmax']:.6g} var_max={m['var_max']:.4g}")
    if "clean" in rep:
        print("=== 未量化对照（客户端本地权重）===")
        for r in rep["clean"]:
            for mod in ("state", "action"):
                if mod not in r:
                    continue
                m = r[mod]
                print(f"  [{r['domain']:<24}] {mod:<6} dim={m['dim']:<3} "
                      f"scale被清零={m['zero_scale_dims']:<3} 有效维度={m['effective_dims']:<3} "
                      f"absmax={m['scale_absmax']:.6g} var_max={m['var_max']:.4g}")

    out = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"[OK] → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
