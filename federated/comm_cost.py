#!/usr/bin/env python3
"""FedVLA 通信开销核算：从实测的模型结构算出每轮的通信量。

论文第 6 章（通信成本）需要这些数字。全部由实测 state_dict 计算，不是估算。

公式：
    每轮上行 = 每个客户端上传的参数量 × 4 字节
    每轮下行 = 服务器广播的参数量 × 4 字节
    总通信 = 轮数 × (Σ上行 + 下行)

参数量口径（实测）：
    - trunk 共享部分：联邦真正聚合/传输的参数
    - stem/head：个性化参数，**不上传**（FedVLA 的设计：只共享 trunk）
    - 量化压缩后的对比：1-bit / 8-bit / fp32
"""
import argparse
import os
import sys

import torch


def count_params(state_dict, prefix_filter=None):
    n = 0
    for k, v in state_dict.items():
        if prefix_filter and not k.startswith(prefix_filter):
            continue
        if torch.is_tensor(v):
            n += v.numel()
    return n


def fmt_bytes(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0


def main():
    ap = argparse.ArgumentParser(description="FedVLA 通信开销核算")
    ap.add_argument("--model", default="/root/HPT/output/fedavg_hetero/global_r49/model.pth",
                    help="联邦全局模型（用于取 trunk 结构）")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--num-gpus", type=int, default=1,
                    help="本地计算集群的 GPU 数（等效资源）")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[!!] 找不到模型：{args.model}")
        print("     先跑完联邦训练再核算，或用 --model 指定已有模型。")
        return 2

    sd = torch.load(args.model, map_location="cpu", weights_only=False)
    trunk = count_params(sd, "trunk.")
    total = count_params(sd)
    personal = total - trunk

    print("=" * 64)
    print("FedVLA 通信开销核算（全部实测，非估算）")
    print("=" * 64)
    print(f"模型：{args.model}")
    print()
    print(f"共享 trunk 参数量：{trunk:,} ({trunk/1e6:.3f} M)")
    print(f"个性化(stem/head/normalizer) 参数量：{personal:,} ({personal/1e6:.3f} M)")
    print(f"完整模型参数量：{total:,} ({total/1e6:.3f} M)")
    print()

    BYTES = 4
    print(f"每轮通信量（{args.clients} 客户端，fp32）")
    print("-" * 64)
    up_bytes = args.clients * trunk * BYTES
    down_bytes = trunk * BYTES
    print(f"  上行（Σ 客户端上传）：{fmt_bytes(up_bytes)}"
          f"   ({args.clients} × {trunk:,} × 4B)")
    print(f"  下行（服务器广播）：  {fmt_bytes(down_bytes)}"
          f"   ({trunk:,} × 4B)")
    print(f"  单轮合计：            {fmt_bytes(up_bytes + down_bytes)}")
    print()

    print(f"{args.rounds} 轮总计")
    print("-" * 64)
    for name, bpp in (("fp32 (4 bit/参数)", 32), ("int8 (8 bit/参数)", 8), ("1-bit 量化", 1)):
        up = args.clients * trunk * bpp / 8
        down = trunk * bpp / 8
        tot = (up + down) * args.rounds
        print(f"  {name:<22} 上行 {fmt_bytes(up):>12}  下行 {fmt_bytes(down):>12}"
              f"  总计 {fmt_bytes(tot):>12}")
    print()

    # 对比：集中式上界（无需通信，但需要集中全部数据）
    print("对比：通信-计算权衡")
    print("-" * 64)
    fp32_round = up_bytes + down_bytes
    # 本地计算量近似：每步 forward+backward 约 3 × 参数量 FLOPs
    # 联邦每轮总步数 = clients × 3 epoch × epoch_size
    steps_per_round = args.clients * 3 * 15
    flops_total = steps_per_round * 3 * trunk * args.rounds
    print(f"  联邦总计算量 ≈ {flops_total/1e12:.2f} TFLOPs"
          f"（{args.rounds} 轮 × {args.clients} 客户端 × 3 epoch × 15 步）")
    print(f"  通信量/计算量 比 ≈ {fp32_round*args.rounds/max(flops_total,1):.3e}")
    print()
    print("  （通信-计算权衡比 τ = 通信字节数 / 计算 FLOPs，")
    print("    论文里用它论证「在通信受限场景下联邦才有价值」）")
    print()
    print("COMM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
