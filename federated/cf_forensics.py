#!/usr/bin/env python3
"""量化归因取证（纯数值，CPU 秒级，不跑 MuJoCo）。

背景
----
第4阶段通信侧出现一个反常现象：**int8 / int4 量化的 task3 成功率全为 0.000**，
而量化自检显示 int8 的权重相对误差只有 0.44% ⇒ 物理上不该把成功率打到 0。

本脚本做三件事，把"到底是量化打坏了什么"钉死：

  Q1 哪些**非零**元素被量化清零？占比多少？集中在哪些张量？
  Q2 把"干净权重"用同一 scheme 再量化一次，零掩码是否与"全局模型"完全一致？
     （一致 ⇒ 归因闭环：清零确实由量化造成，不是别的原因）
  Q3 normalizer 的 scale / offset 有多少维度被清零？
     ⇒ 这些维度上的输入信息被彻底抹掉（x*0+offset ≡ 常数）。

对照口径
--------
  clean = 客户端**本地**权重（client_i_r49/model.pth）
          量化发生在服务端聚合（aggregate.py 里 q.apply_tensor），
          客户端本地权重**未经量化** ⇒ 它是干净参照。
  quant = 聚合后的**全局**模型（global_r49/model.pth）⇒ 含量化损伤。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, "/root/fedcore")
from comm_analysis import quantize_per_group  # noqa: E402  （与链路同源实现）

BITS = {"int8_g": 8, "int4_g": 4, "bit1": 1}


def as_sd(obj):
    """摊平成 {key: tensor}（支持一层嵌套）。"""
    sd = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if torch.is_tensor(v):
                sd[str(k)] = v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if torch.is_tensor(v2):
                        sd[f"{k}.{k2}"] = v2
    return sd


def flat(d):
    """{文件名::key: float tensor}"""
    out = {}
    for fn, obj in d.items():
        for k, t in as_sd(obj).items():
            if torch.is_tensor(t) and t.is_floating_point():
                out[f"{fn}::{k}"] = t.detach().float()
    return out


def load_dir(d):
    objs = {}
    for f in sorted(glob.glob(os.path.join(d, "*.pth")) + glob.glob(os.path.join(d, "*.pt"))):
        try:
            objs[os.path.basename(f)] = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读不了 {f}: {type(e).__name__}: {e}")
    return objs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="客户端本地权重目录（干净参照）")
    ap.add_argument("--quant", required=True, help="全局模型目录（含量化损伤）")
    ap.add_argument("--scheme", default="int4_g")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    bits = BITS[args.scheme]
    clean = flat(load_dir(args.clean))
    quant = flat(load_dir(args.quant))
    print(f"[i] clean 张量数={len(clean)}  quant 张量数={len(quant)}  scheme={args.scheme} bits={bits}")

    rep = {"scheme": args.scheme, "bits": bits,
           "clean_dir": args.clean, "quant_dir": args.quant}

    # ---------- Q1：谁被清零 ----------
    rows = []
    nz_tot = killed_tot = 0
    for k, q in quant.items():
        c = clean.get(k)
        if c is None or c.shape != q.shape:
            continue
        nz = int((c != 0).sum())
        killed = int(((c != 0) & (q == 0)).sum())
        nz_tot += nz
        killed_tot += killed
        rel = float(torch.linalg.norm(c - q) / (torch.linalg.norm(c) + 1e-12))
        rows.append({"key": k, "shape": list(c.shape),
                     "zeros_clean": int((c == 0).sum()), "zeros_quant": int((q == 0).sum()),
                     "nonzero_clean": nz, "killed": killed,
                     "killed_pct": (killed / nz * 100.0) if nz else 0.0,
                     "rel_err": rel})
    rep["q1_global_nonzero"] = nz_tot
    rep["q1_global_killed"] = killed_tot
    rep["q1_global_killed_pct"] = (killed_tot / nz_tot * 100.0) if nz_tot else 0.0
    rows.sort(key=lambda r: (-r["killed"], r["key"]))
    rep["q1_top_killed"] = rows[: args.top]
    print(f"[Q1] 全局非零元素 {nz_tot}，被清零 {killed_tot} "
          f"({rep['q1_global_killed_pct']:.3f}%)")
    for r in rows[: args.top]:
        print(f"      {r['key']:<70} shape={r['shape']} killed={r['killed']}/{r['nonzero_clean']}"
              f" ({r['killed_pct']:.1f}%) rel={r['rel_err']:.5f}")

    # ---------- Q2：零掩码复现（归因闭环） ----------
    repro = []
    mask_all_same = True
    for k, c in clean.items():
        q = quant.get(k)
        if q is None or q.shape != c.shape:
            continue
        cq = quantize_per_group(c, bits, 64).reshape(-1)
        qf = q.reshape(-1)
        same = bool(((cq == 0) == (qf == 0)).all())
        maxdiff = float((cq - qf).abs().max())
        # 量级一致性：差值是否都在量化步长以内（说明只差聚合噪声）
        if "normalizer" in k:
            repro.append({"key": k, "mask_identical": same,
                          "zeros_repro": int((cq == 0).sum()),
                          "zeros_global": int((qf == 0).sum()),
                          "max_abs_diff": maxdiff})
        if not same:
            mask_all_same = False
    rep["q2_mask_all_identical"] = mask_all_same
    rep["q2_normalizer_repro"] = repro
    print(f"[Q2] 全部同形状张量的零掩码是否完全复现：{mask_all_same}")
    for r in repro:
        print(f"      {r['key']:<70} mask一致={r['mask_identical']} "
              f"复现零数={r['zeros_repro']} 全局零数={r['zeros_global']} maxdiff={r['max_abs_diff']:.6g}")

    # ---------- Q3：normalizer 信息是否被抹掉 ----------
    nrm = []
    for k, c in clean.items():
        if "normalizer" not in k:
            continue
        q = quant.get(k)
        if q is None or q.shape != c.shape:
            continue
        s_clean = c.reshape(-1)
        s_quant = q.reshape(-1)
        lost = int(((s_clean != 0) & (s_quant == 0)).sum())
        nrm.append({"key": k, "dim": int(s_clean.numel()),
                    "lost_dims": lost,
                    "lost_frac": lost / max(int(s_clean.numel()), 1),
                    "absmax_clean": float(s_clean.abs().max()),
                    "min_nonzero_clean": float(s_clean[s_clean != 0].abs().min()) if int((s_clean != 0).sum()) else 0.0,
                    "absmax_quant": float(s_quant.abs().max())})
    rep["q3_normalizer"] = nrm
    print("[Q3] normalizer 各张量被抹掉的维度：")
    for r in nrm:
        print(f"      {r['key']:<70} dim={r['dim']} 丢失={r['lost_dims']} "
              f"({r['lost_frac'] * 100:.1f}%) absmax_clean={r['absmax_clean']:.6g} "
              f"min_nonzero={r['min_nonzero_clean']:.6g}")

    # ---------- 附加：normalizer 输出的常数性 ----------
    # 若 scale 的某维为 0 ⇒ 归一化输出该维恒为 offset ⇒ 该维输入信息彻底消失。
    const = []
    for r in nrm:
        if "scale" not in r["key"]:
            continue
        k = r["key"]
        c = clean[k].reshape(-1)
        q = quant[k].reshape(-1)
        dead = (c != 0) & (q == 0)
        const.append({"key": k, "dead_dims": int(dead.sum()),
                      "example_true_values": [round(float(v), 6) for v in c[dead][:8]],
                      "example_quant_values": [round(float(v), 6) for v in q[dead][:8]]})
    rep["q3_dead_dims"] = const
    print("[Q3b] 被抹掉的维度（真值 → 量化后）：")
    for r in const:
        print(f"      {r['key']:<70} dead={r['dead_dims']} {r['example_true_values']} → {r['example_quant_values']}")

    out = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[OK] 已写出 {args.out}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
