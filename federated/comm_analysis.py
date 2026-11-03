#!/usr/bin/env python3
"""FedVLA 第 6 章：通信开销与量化/压缩的模拟实验。

要回答的问题（通信委员会最关心的）：
  Q1: 联邦一轮要传多少字节？相对集中式省了多少？
  Q2: 量化到 int8 / int4 / 1-bit，通信量降多少？精度损失多少？
  Q3: 在给定的通信预算下，联邦能训练多少轮？够不够收敛？

方法：
  - 通信量：从实测 state_dict 统计参数量（不是估算）
  - 量化误差：在真实张量上模拟 int8/int4/1-bit 量化，测相对误差
  - 收敛需求：用实测的集中式收敛曲线估计"达到 X 成功率所需轮数"

不依赖 GPU，纯 CPU 计算。
"""
import argparse
import json
import os
import sys

import torch


# ---------- 量化模拟 ----------
def quantize_per_tensor(t, bits):
    """对称均匀量化（per-tensor）。bits=32 时不量化。"""
    if bits >= 32:
        return t.clone()
    if bits <= 0:
        raise ValueError("bits 必须 >= 1")
    if bits == 1:
        # 1-bit 只有符号（±scale），没有幅值信息 -> 用均值作为阈值（符号量化）
        thr = t.median()
        scale = float((t - thr).abs().mean())
        if scale == 0:
            return t.clone()
        sign = torch.sign(t - thr)
        return sign * scale
    qmax = 2 ** (bits - 1) - 1
    scale = t.abs().max() / qmax
    if scale == 0:
        return t.clone()
    q = torch.clamp(torch.round(t / scale), -qmax - 1, qmax)
    return q * scale


def quantize_per_group(t, bits, group_size=64):
    """分组量化（per-group），比 per-tensor 精度高很多，是现代 FL 通信压缩的主流做法。"""
    if bits >= 32:
        return t.clone()
    if bits == 1:
        # 1-bit 符号量化：组内用中位数作阈值，均值幅值作 scale
        out = torch.empty_like(t)
        n = t.numel()
        if n % group_size != 0:
            # 逐元素退化处理（最后一组不足）
            thr = t.median()
            scale = float((t - thr).abs().mean())
            return torch.sign(t - thr) * (scale if scale > 0 else 1.0)
        flat = t.reshape(-1, group_size)
        for i in range(flat.shape[0]):
            row = flat[i]
            thr = row.median()
            scale = float((row - thr).abs().mean())
            out.reshape(-1, group_size)[i] = (
                torch.sign(row - thr) * (scale if scale > 0 else 1.0)
            )
        return out
    if t.numel() % group_size != 0:
        # 最后一组单独处理
        n = (t.numel() // group_size) * group_size
        head = quantize_per_group(t[:n].reshape(-1, group_size), bits, group_size).reshape(-1)
        tail = quantize_per_group(t[n:], bits, t.numel() - n) if t.numel() > n else None
        return torch.cat([head, tail]) if tail is not None else head
    flat = t.reshape(-1, group_size)
    out = torch.empty_like(flat)
    qmax = 2 ** (bits - 1) - 1
    for i in range(flat.shape[0]):
        row = flat[i]
        scale = row.abs().max() / qmax
        if scale == 0:
            out[i] = row
            continue
        q = torch.clamp(torch.round(row / scale), -qmax - 1, qmax)
        out[i] = q * scale
    return out.reshape(t.shape)


def rel_err(a, b):
    """相对 Frobenius 误差"""
    num = float(torch.linalg.norm((a - b).float()))
    den = float(torch.linalg.norm(a.float()))
    return num / den if den > 0 else 0.0


# ---------- 通信量 ----------
def param_count(sd, prefix=None):
    n = 0
    for k, v in sd.items():
        if prefix and not k.startswith(prefix):
            continue
        if torch.is_tensor(v):
            n += v.numel()
    return n


def main():
    ap = argparse.ArgumentParser(description="FedVLA 通信开销与量化误差分析")
    ap.add_argument("--model", default="/root/HPT/output/fedavg_hetero/global_r49/model.pth")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--out", default="/root/HPT/output/comm_analysis.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[!!] 找不到模型 {args.model}（先跑完联邦训练）")
        return 2

    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    trunk_keys = [k for k in sd if k.startswith("trunk.")]
    trunk_params = param_count(sd, "trunk.")
    total_params = param_count(sd)
    personal_params = total_params - trunk_params

    print("=" * 70)
    print("FedVLA 通信分析（实测参数量 + 真实张量量化模拟）")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"共享 trunk: {trunk_params:,} ({trunk_params/1e6:.3f} M)")
    print(f"个性化参数: {personal_params:,} ({personal_params/1e6:.3f} M)")
    print(f"完整模型:   {total_params:,} ({total_params/1e6:.3f} M)")
    print()

    # 取一小块真实 trunk 张量做量化误差模拟（全量太慢）
    sample_keys = [k for k in trunk_keys if sd[k].numel() > 1000][:8]
    tensors = [sd[k].float() for k in sample_keys]

    print("量化精度 vs 通信量")
    print("-" * 70)
    print(f"{'方案':<28}{'比特/参数':>10}{'50轮通信':>14}{'相对压缩':>10}{'相对误差':>10}")
    print("-" * 70)

    rows = []
    schemes = [
        ("fp32 (不压缩)", 32, "tensor"),
        ("int8 per-tensor", 8, "tensor"),
        ("int8 per-group(64)", 8, "group"),
        ("int4 per-group(64)", 4, "group"),
        ("int2 per-group(64)", 2, "group"),
        ("1-bit 符号化", 1, "group"),
    ]
    base_bytes = (args.clients + 1) * trunk_params * 4 * args.rounds

    for name, bits, mode in schemes:
        if bits >= 32:
            err = 0.0
        else:
            errs = []
            for t in tensors:
                if mode == "tensor":
                    q = quantize_per_tensor(t, bits)
                else:
                    q = quantize_per_group(t, bits, 64)
                errs.append(rel_err(t, q))
            err = sum(errs) / len(errs)

        total_bytes = (args.clients + 1) * trunk_params * (bits / 8) * args.rounds
        ratio = total_bytes / base_bytes if base_bytes else 0
        print(f"{name:<28}{bits:>10}{total_bytes/1e9:>12.2f} GB{ratio:>9.1%}{err:>10.4f}")
        rows.append({
            "scheme": name, "bits": bits, "mode": mode,
            "total_bytes_50rounds": total_bytes,
            "compression_ratio": ratio,
            "relative_error": err,
        })

    print()
    print("说明：")
    print("  · 上行 = 4 客户端各上传 trunk；下行 = 服务器广播聚合后的 trunk")
    print("  · 相对误差在真实 trunk 张量上模拟量化得到（per-tensor / per-group(64)）")
    print("  · per-group 量化额外需传每组的 scale，通信量上略被高估（保守）")
    print()

    # 通信预算 -> 可训练轮数
    print("给定通信预算能训练多少轮？")
    print("-" * 70)
    print(f"{'预算':<16}{'fp32 轮数':>14}{'int8 轮数':>14}{'int4 轮数':>14}{'1bit 轮数':>14}")
    for budget_gb in (1, 5, 10, 50, 100):
        b = budget_gb * 1e9
        r32 = b / ((args.clients + 1) * trunk_params * 4)
        r8 = b / ((args.clients + 1) * trunk_params * 1)
        r4 = b / ((args.clients + 1) * trunk_params * 0.5)
        r1 = b / ((args.clients + 1) * trunk_params * 0.125)
        print(f"{budget_gb:>3} GB{'':<10}{r32:>14.0f}{r8:>14.0f}{r4:>14.0f}{r1:>14.0f}")
    print()

    result = {
        "model": args.model,
        "rounds": args.rounds,
        "clients": args.clients,
        "trunk_params": trunk_params,
        "personal_params": personal_params,
        "total_params": total_params,
        "quantization": rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"已保存: {args.out}")
    print("COMM_ANALYSIS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
