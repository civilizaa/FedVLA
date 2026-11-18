#!/usr/bin/env python
"""上行量化链路的离线单测（2026-11-18 新增）。

为什么必须写这个单测：
  第4阶段通信扫描实测 int8_g / int4_g / bit1 三档成功率**全为 0.000**，
  而离线在真实模型上模拟 int8_g 的相对误差只有 0.005。
  0.5% 的权重扰动不可能把策略从 0.75 打到 0.00 ⇒ 矛盾。
  本单测就是用来判定："量化数学本身"能不能解释这个归零。
  如果本单测通过（量化→聚合等价于原聚合，误差 ~1e-3 量级），
  那归零**必然是接线/统计口径 bug**，而不是信道现象 —— 结论必须改写。

跑法（Windows 本机，CPU 即可）：
    "D:/conda_envs/lerobot/python.exe" federated/test_quant_channel.py
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from comm_analysis import quantize_per_group, rel_err  # noqa: E402

GROUP = 64
FAIL = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def _fake_shared_state(n_blocks: int = 4, d: int = 512) -> dict:
    """造一份形似 HPT trunk 的共享 state_dict（键前缀 trunk.）。

    真实权重的尺度（LayerNorm 权重 ~1.0、线性层权重 ~0.02）都覆盖到，
    因为 per-group 量化的 scale=max/127 对**离群值**最敏感 —— 尺度不真实会低估误差。
    """
    sd = {}
    g = torch.Generator().manual_seed(20261118)
    for b in range(n_blocks):
        sd[f"trunk.blocks.{b}.attn.qkv.weight"] = torch.randn(d, d, generator=g) * 0.02
        sd[f"trunk.blocks.{b}.attn.qkv.bias"] = torch.randn(d, generator=g) * 0.01
        sd[f"trunk.blocks.{b}.ln1.weight"] = torch.ones(d) * 1.0
        sd[f"trunk.blocks.{b}.ln1.bias"] = torch.zeros(d)
        sd[f"trunk.blocks.{b}.mlp.fc1.weight"] = torch.randn(d, d * 2, generator=g) * 0.02
    return sd


def test_scheme_errors() -> None:
    print("\n[1] 各量化方案的相对误差（per-group, group=64）")
    sd = _fake_shared_state()
    flat = torch.cat([v.reshape(-1) for v in sd.values()])
    for scheme, bits, ceiling in (("int8_g", 8, 0.02), ("int4_g", 4, 0.25),
                                  ("int2_g", 2, 0.90), ("bit1", 1, 1.20)):
        q = quantize_per_group(flat, bits, GROUP)
        ok_nan = not (torch.isnan(q).any() or torch.isinf(q).any())
        e = rel_err(flat, q)
        _check(f"{scheme} 无 NaN/Inf", ok_nan)
        _check(f"{scheme} 相对误差 {e:.5f} <= {ceiling}", e <= ceiling)


def test_aggregate_equivalence() -> None:
    """决定性判据：量化后再加权平均，是否还等价于原来的加权平均。

    通信扫描真正执行的是「客户端模型 → 量化 → 服务器加权平均」。
    如果这一步的相对偏差在 1e-3 ~ 1e-2 量级，那 int8_g 不可能导致成功率归零。
    """
    print("\n[2] 量化→聚合 等价性（服务器加权平均）")
    g = torch.Generator().manual_seed(7)
    clients = []
    for _ in range(4):
        sd = _fake_shared_state()
        # 模拟"各客户端训练后的差异"：加一点扰动
        for k in sd:
            sd[k] = sd[k] + torch.randn(sd[k].shape, generator=g) * 0.005
        clients.append(sd)
    weights = [0.25, 0.25, 0.25, 0.25]

    def avg(sds, w):
        out = {}
        for k in sds[0]:
            acc = torch.zeros_like(sds[0][k], dtype=torch.float64)
            for si, sd in enumerate(sds):
                acc += w[si] * sd[k].double()
            out[k] = acc.float()
        return out

    ref = avg(clients, weights)
    # int8/int4 必须**极小**（<5%）⇒ 量化数学无法解释成功率归零；
    # bit1 是"重破坏"区间（>30%）⇒ 它的归零可以是真实现象。
    for scheme, bits, ceiling in (("int8_g", 8, 0.05), ("int4_g", 4, 0.10)):
        qcl = []
        for sd in clients:
            qsd = {k: quantize_per_group(v, bits, GROUP) for k, v in sd.items()}
            qcl.append(qsd)
        got = avg(qcl, weights)
        num = sum(float(torch.linalg.norm((got[k] - ref[k]).double())) for k in ref)
        den = sum(float(torch.linalg.norm(ref[k].double())) for k in ref)
        d = num / den if den else 0.0
        _check(f"{scheme} 聚合后相对偏差 {d:.6f} <= {ceiling}", d <= ceiling,
               "（越小说明量化不可能是成功率归零的原因）")
    qcl = [{k: quantize_per_group(v, 1, GROUP) for k, v in sd.items()} for sd in clients]
    got = avg(qcl, weights)
    num = sum(float(torch.linalg.norm((got[k] - ref[k]).double())) for k in ref)
    den = sum(float(torch.linalg.norm(ref[k].double())) for k in ref)
    d1 = num / den if den else 0.0
    _check(f"bit1 聚合后相对偏差 {d1:.6f} 落在重破坏区间 (0.3~1.2)", 0.3 <= d1 <= 1.2,
           "（1-bit 本就可能摧毁策略 ⇒ 其 0.000 属真实效应）")


def test_personal_params_untouched() -> None:
    """stems./heads. 是个性化参数，不上行 ⇒ 不能被量化。"""
    print("\n[3] 个性化参数不参与量化")
    from aggregate import _is_shared_param
    _check("stems.x 被判为非共享", not _is_shared_param("stems.reach-v2.weight"))
    _check("heads.x 被判为非共享", not _is_shared_param("heads.reach-v2.weight"))
    _check("trunk.x 被判为共享", _is_shared_param("trunk.blocks.0.attn.qkv.weight"))
    _check("normalizer 被判为共享", _is_shared_param("normalizer.reach-v2.mean"))


def test_nan_detected() -> None:
    """自检必须能抓到 NaN —— 这是本轮要加的核心闸。"""
    print("\n[4] NaN 检测（模拟 aggregate.py 的量化自检）")
    sd = _fake_shared_state(n_blocks=1)
    sd["trunk.blocks.0.mlp.fc1.weight"][0, 0] = float("nan")
    qsd = {k: quantize_per_group(v, 8, GROUP) for k, v in sd.items()}
    bad = [k for k, v in qsd.items() if torch.isnan(v).any() or torch.isinf(v).any()]
    _check("NaN 被检出", len(bad) > 0, f"检出 {len(bad)} 个键")
    if not bad:
        FAIL.append("NaN 检出")


def _make_client_dirs(tmp: str, n: int = 2, inject_nan: bool = False) -> list:
    """造 n 个客户端目录，各含一个 model.pth（形似 HPT 的 state_dict）。"""
    dirs = []
    for i in range(n):
        d = os.path.join(tmp, f"c{i}")
        os.makedirs(d, exist_ok=True)
        g = torch.Generator().manual_seed(100 + i)
        sd = {
            "trunk.trunk.blocks.0.attn.qkv.weight": torch.randn(256, 256, generator=g) * 0.02,
            "trunk.trunk.blocks.0.ln1.weight": torch.ones(256),
            "trunk.trunk.blocks.0.ln1.bias": torch.zeros(256),
            "normalizer.reach-v2.mean": torch.randn(32, generator=g),
            "heads.reach-v2.fc.weight": torch.randn(4, 32, generator=g) * 0.01,
            "stems.reach-v2.conv.weight": torch.randn(4, 3, 3, 3, generator=g) * 0.01,
        }
        if inject_nan and i == 0:
            sd["trunk.trunk.blocks.0.attn.qkv.weight"][0, 0] = float("nan")
        torch.save(sd, os.path.join(d, "model.pth"))
        dirs.append(d)
    return dirs


def test_aggregate_e2e() -> None:
    """端到端跑 aggregate.py 的量化分支：自检 + 漂移诊断 + NaN 必须中止。"""
    print("\n[5] aggregate.py 量化分支 端到端")
    import shutil
    import tempfile

    from aggregate import aggregate

    tmp = tempfile.mkdtemp(prefix="fedvla_quant_")
    try:
        dirs = _make_client_dirs(tmp, n=2)
        out = os.path.join(tmp, "global")
        aggregate(dirs, out, quant_scheme="int8_g")
        _check("int8_g 聚合成功", os.path.exists(os.path.join(out, "model.pth")))
        dj = os.path.join(out, "quant_drift.jsonl")
        _check("漂移诊断文件已写出", os.path.exists(dj))
        if os.path.exists(dj):
            with open(dj, encoding="utf-8") as fh:
                lines = [l for l in fh.read().splitlines() if l.strip()]
            _check("漂移诊断有 1 行记录", len(lines) == 1, f"{len(lines)} 行")
            if lines:
                import json as _json
                rec = _json.loads(lines[0])
                _check(f"漂移值 {rec.get('drift'):.6f} < 0.05", rec.get("drift", 9) < 0.05)
    except Exception as e:  # noqa: BLE001
        _check("int8_g 聚合成功", False, f"异常：{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # NaN 客户端必须被自检拦下（抛错），而不是静默产出 0 分曲线
    tmp2 = tempfile.mkdtemp(prefix="fedvla_quant_nan_")
    try:
        dirs = _make_client_dirs(tmp2, n=2, inject_nan=True)
        out2 = os.path.join(tmp2, "global")
        raised = False
        try:
            aggregate(dirs, out2, quant_scheme="int8_g")
        except ValueError as e:
            raised = True
            print(f"       ↳ 正确拦截：{str(e)[:80]}")
        except Exception as e:  # noqa: BLE001
            raised = True
            print(f"       ↳ 拦截（异常类型 {type(e).__name__}）：{str(e)[:80]}")
        _check("NaN 客户端被量化自检拦下", raised)
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)


def main() -> int:
    print("=" * 66)
    print("上行量化链路离线单测")
    print("=" * 66)
    test_scheme_errors()
    test_aggregate_equivalence()
    test_personal_params_untouched()
    test_nan_detected()
    test_aggregate_e2e()
    print("\n" + "=" * 66)
    if FAIL:
        print(f"❌ {len(FAIL)} 项失败：" + "; ".join(FAIL))
        return 1
    print("✅ 全部通过")
    print("\n结论提要：若第 2 项（量化→聚合 等价性）通过，")
    print("则 int8_g/int4_g/bit1 成功率归零**不能**由量化数学解释，")
    print("必须是接线/统计口径故障 —— 论文结论必须改写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
