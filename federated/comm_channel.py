#!/usr/bin/env python
"""FedVLA 第4阶段：联邦上行通信链路模拟（丢包 + 量化压缩）。

用途
----
在"客户端本地训练完成"与"服务器聚合"之间插入一条**受损伤的通信链路**，
用来测量链路损伤对最终任务成功率的影响（第4阶段的三条曲线）。

支持两类损伤
------------
1) **客户端级丢包**（client-level dropout）
   整个客户端的上行更新丢失 → 服务器本轮不收到它的贡献。
   这是 FL 里最常见、最容易被委员会理解的模型（设备掉线 / 信道中断）。
   注意：客户端被丢弃时，它独占域的 stem/head 也会随之缺席，
   必须在服务器端回退到**上一轮全局模型**里的对应参数（见 aggregate.py
   的 --prev-global），否则全局模型缺域 → 评估直接崩。

2) **上行量化压缩**（per-group 对称量化）
   模拟"把模型权重按更少的比特发出去，服务器收到后反量化"。
   量化对**所有浮点张量**生效（trunk 权重 + stem/head + normalizer），
   因为真实链路不会区分"哪些参数更该压缩"。

⚠️ **口径统一（重要）**
   量化函数与误差口径**直接复用 `comm_analysis.py`**（门2 用的同一个脚本），
   保证第4章「通信成本表」与第4章「通信损伤曲线」两处的数字可以互相印证。
   若在这里另写一套量化实现，两张表的误差值会对不上，评审一眼就能看出来。

量化方案
--------
| scheme      | 每参数比特 | 说明                                     |
|-------------|-----------|------------------------------------------|
| fp32        | 32        | 基线（不压缩）                            |
| fp16        | 16        | 半精度                                    |
| int8_g      | 8         | per-group int8，group=64（门2 同款）       |
| int4_g      | 4         | per-group int4，group=64（门2 同款）       |
| bit1        | 1         | per-group 符号化（中位数阈值，门2 同款）    |
| sparse1pct  | 稀疏      | 只传绝对值最大的 1% 参数（新增方案）        |

字节统计口径与 comm_analysis.py 一致：`numel * bits/8`（忽略 per-group scale
的额外开销 —— 门2 同样忽略，注释里已声明这是保守估计）。

用法
----
命令行：
    python comm_channel.py quantize --scheme int8_g --in a.pth --out b.pth
    python comm_channel.py stats    --in a.pth --scheme int8_g
作为库：
    from comm_channel import Quantizer, simulate_uplink, drop_clients
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable

import torch

# 与 comm_analysis.py 共用同一套量化实现与误差口径（见文件头说明）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comm_analysis import quantize_per_group, quantize_per_tensor, rel_err  # noqa: E402


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SCHEMES = {
    "fp32":       {"bits": 32,   "mode": "none"},
    "fp16":       {"bits": 16,   "mode": "cast"},
    "int8_g":     {"bits": 8,    "mode": "group"},
    "int4_g":     {"bits": 4,    "mode": "group"},
    "int2_g":     {"bits": 2,    "mode": "group"},
    "bit1":       {"bits": 1,    "mode": "group"},
    "sparse1pct": {"bits": None, "mode": "sparse"},
}

DEFAULT_GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# 量化器
# ---------------------------------------------------------------------------

def _is_float_tensor(t: torch.Tensor) -> bool:
    return t.is_floating_point()


def _quantize_sparse(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """只保留绝对值最大的 keep_ratio 比例元素，其余置零（模拟稀疏上行）。"""
    n = x.numel()
    if n == 0 or keep_ratio >= 1.0:
        return x.clone()
    k = max(1, int(math.ceil(n * keep_ratio)))
    flat = x.reshape(-1)
    thresh = flat.abs().kthvalue(n - k + 1).values
    out = torch.where(flat.abs() >= thresh, flat, torch.zeros_like(flat))
    return out.reshape(x.shape)


class Quantizer:
    """把一个 state_dict 按指定 scheme 做「量化 → 反量化」，模拟链路接收到的值。"""

    def __init__(self, scheme: str = "fp32", group_size: int = DEFAULT_GROUP_SIZE,
                 keep_ratio: float = 0.01):
        if scheme not in SCHEMES:
            raise ValueError(f"未知 scheme: {scheme}（可选：{list(SCHEMES)}）")
        self.scheme = scheme
        self.group_size = max(1, int(group_size))
        self.keep_ratio = float(keep_ratio)
        self.spec = SCHEMES[scheme]

    # -- 核心 ---------------------------------------------------------------

    def apply_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.spec["mode"]
        if mode == "none" or not _is_float_tensor(x):
            return x.clone()
        if mode == "cast":
            # fp16：转成 fp16 再转回（真实链路就是这个过程）
            return x.to(torch.float16).to(x.dtype)
        if mode == "group":
            return quantize_per_group(x, self.spec["bits"], self.group_size)
        if mode == "sparse":
            return _quantize_sparse(x, self.keep_ratio)
        return x.clone()

    def apply_state_dict(self, sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            out[k] = self.apply_tensor(v) if torch.is_tensor(v) else v
        return out

    # -- 误差（与 comm_analysis.py 同口径：按张量取 rel_err 后平均）-----------

    def mean_tensor_rel_err(self, sd: dict) -> float:
        """按张量平均的相对 Frobenius 误差 —— 与门2 的表同口径。"""
        errs = []
        for v in sd.values():
            if not torch.is_tensor(v) or not _is_float_tensor(v):
                continue
            q = self.apply_tensor(v)
            errs.append(rel_err(v, q))
        return (sum(errs) / len(errs)) if errs else 0.0

    def global_rel_err(self, sd: dict, qsd: dict) -> float:
        """全局聚合的相对误差（把整份 state_dict 拼成一个 Frobenius 范数）。
        与 mean_tensor_rel_err 互补：前者反映"整体偏离"，后者反映"平均张量偏离"。"""
        num = 0.0
        den = 0.0
        for k, v in sd.items():
            if not torch.is_tensor(v) or not _is_float_tensor(v):
                continue
            w = qsd.get(k)
            if w is None or not torch.is_tensor(w):
                continue
            num += float(torch.linalg.norm((v.float() - w.float())))
            den += float(torch.linalg.norm(v.float()))
        return (num / den) if den > 0 else 0.0

    # -- 通信量（与 comm_analysis.py 同口径：numel * bits/8）-----------------

    def theoretical_bytes(self, sd: dict) -> int:
        """理论上行字节数。浮点张量按 numel*bits/8；非浮点按 4 字节保守估计。"""
        total = 0
        bits = self.spec["bits"]
        for v in sd.values():
            if not torch.is_tensor(v):
                continue
            n = v.numel()
            if not _is_float_tensor(v):
                total += n * 4
            elif bits is None:                      # sparse
                k = max(1, int(math.ceil(n * self.keep_ratio)))
                total += k * 8 + 16                # 索引(4B) + 值(4B) + 元信息
            else:
                total += n * bits // 8
        return int(total)

    def compression_ratio(self, sd: dict) -> float:
        """相对 fp32 的压缩比（<1 表示更小）。"""
        raw = 0
        for v in sd.values():
            if torch.is_tensor(v):
                raw += v.numel() * 4
        if raw == 0:
            return 1.0
        return self.theoretical_bytes(sd) / raw


# ---------------------------------------------------------------------------
# 丢包
# ---------------------------------------------------------------------------

def drop_clients(n_clients: int, drop_rate: float, seed: int = 0,
                 always_keep_first: bool = True, mode: str = "fixed",
                 round_idx: int = 0) -> list[int]:
    """返回本轮被丢弃的客户端索引列表。

    mode
    ----
    - "fixed"（默认，推荐做曲线用）
      每轮**固定**丢弃 k = floor(n_clients × drop_rate) 个客户端，
      被丢的客户端按轮次轮换，保证 50 轮里每个客户端被丢的次数尽量均等。
      好处：损伤量**每轮恒定** → 曲线单调可解释，且不同配置之间可比。

    - "bernoulli"
      每一轮独立地按概率丢每个客户端（真实信道的统计模型）。
      缺点：在 n_clients=4、drop_rate=0.1 时，期望每轮只丢 0.4 个客户端，
      大量轮次一个都不丢 → 损伤被随机性稀释，曲线几乎看不出差异。
      实测 drop_rate=0.1 / seed=7 出现"连续多轮零丢弃"，故不作为主口径。

    always_keep_first: 永远保留第 0 个客户端。
    理由：若某轮所有客户端都被丢弃，聚合无从进行（除数 0 / 全局模型缺域），
    实验会出现"一轮都没训"的空转。工程上真实系统通常也会保底留一路。
    """
    if drop_rate <= 0:
        return []
    if mode not in ("fixed", "bernoulli"):
        raise ValueError(f"未知 drop mode: {mode}")

    if mode == "bernoulli":
        g = torch.Generator().manual_seed(int(seed))
        r = torch.rand(n_clients, generator=g).tolist()
        dropped = [i for i in range(n_clients) if r[i] < drop_rate]
        if always_keep_first and 0 in dropped and len(dropped) == n_clients:
            dropped.remove(0)
        return dropped

    # --- fixed：每轮固定 k 个，轮换覆盖 ---
    k = int(math.floor(n_clients * drop_rate))
    if k <= 0:
        return []
    if k >= n_clients:
        k = max(1, n_clients - 1)          # 至少留一路
    # 用 (seed, round_idx) 决定本轮起始偏移，保证轮换确定且可复现
    offset = (int(seed) * 7919 + int(round_idx)) % n_clients
    order = [(offset + i) % n_clients for i in range(n_clients)]
    dropped = order[:k]
    if always_keep_first and 0 in dropped and len(dropped) == n_clients:
        dropped.remove(0)
    return sorted(dropped)


# ---------------------------------------------------------------------------
# 组合：模拟一次上行传输
# ---------------------------------------------------------------------------

def simulate_uplink(client_sd: dict, scheme: str = "fp32",
                    group_size: int = DEFAULT_GROUP_SIZE,
                    keep_ratio: float = 0.01) -> dict:
    """单个客户端的上行更新经过链路后的样子（量化→反量化）。

    返回 {'state': dict, 'bytes': int, 'ratio': float, 'rel_err': float}
    """
    q = Quantizer(scheme, group_size=group_size, keep_ratio=keep_ratio)
    qsd = q.apply_state_dict(client_sd)
    return {
        "state": qsd,
        "bytes": q.theoretical_bytes(client_sd),
        "ratio": q.compression_ratio(client_sd),
        "rel_err": q.mean_tensor_rel_err(client_sd),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_quantize(args) -> None:
    sd = torch.load(args.inp, map_location="cpu", weights_only=False)
    q = Quantizer(args.scheme, group_size=args.group_size, keep_ratio=args.keep_ratio)
    out = q.apply_state_dict(sd)
    torch.save(out, args.out)
    print(json.dumps({
        "scheme": args.scheme,
        "mean_tensor_rel_err": q.mean_tensor_rel_err(sd),
        "global_rel_err": q.global_rel_err(sd, out),
        "bytes": q.theoretical_bytes(sd),
        "ratio": q.compression_ratio(sd),
        "out": args.out,
    }, ensure_ascii=False))


def _cmd_stats(args) -> None:
    sd = torch.load(args.inp, map_location="cpu", weights_only=False)
    rows = []
    for name in (args.schemes.split(",") if args.schemes else list(SCHEMES)):
        q = Quantizer(name, group_size=args.group_size, keep_ratio=args.keep_ratio)
        qsd = q.apply_state_dict(sd)
        rows.append({
            "scheme": name,
            "bits": q.spec["bits"],
            "mean_tensor_rel_err": q.mean_tensor_rel_err(sd),
            "global_rel_err": q.global_rel_err(sd, qsd),
            "bytes": q.theoretical_bytes(sd),
            "mib": q.theoretical_bytes(sd) / (1024 * 1024),
            "ratio": q.compression_ratio(sd),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def _cmd_drop(args) -> None:
    print(json.dumps(drop_clients(args.n, args.rate, args.seed,
                                  mode=args.mode, round_idx=args.round),
                     ensure_ascii=False))


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="FedVLA 上行通信链路模拟")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("quantize", help="量化一个 state_dict 并落盘")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--scheme", default="fp32", choices=list(SCHEMES))
    q.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    q.add_argument("--keep-ratio", type=float, default=0.01)
    q.set_defaults(func=_cmd_quantize)

    s = sub.add_parser("stats", help="比较各量化方案的误差与通信量")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--schemes", default="")
    s.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    s.add_argument("--keep-ratio", type=float, default=0.01)
    s.set_defaults(func=_cmd_stats)

    d = sub.add_parser("drop", help="按丢包率抽客户端")
    d.add_argument("--n", type=int, required=True)
    d.add_argument("--rate", type=float, required=True)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--mode", default="fixed", choices=["fixed", "bernoulli"],
                   help="fixed=每轮固定丢 k 个并轮换（推荐）；bernoulli=独立随机")
    d.add_argument("--round", type=int, default=0, dest="round",
                   help="轮次（fixed 模式用它决定轮换偏移）")
    d.set_defaults(func=_cmd_drop)

    args = p.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
