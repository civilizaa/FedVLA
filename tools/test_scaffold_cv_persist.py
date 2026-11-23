#!/usr/bin/env python3
"""SCAFFOLD 本地控制变量 c_k 落盘路径的离线合成验证（铁律 E：新/疑点代码先离线单测）。

复现云端 2026-11-23 的现象：p5_bal_scaffold 在第 1 轮聚合时报
    「客户端 0 缺本地控制变量 .../_baseline/ci_c0_r1.pt —— SCAFFOLD 聚合中止」
⇒ 说明 hook 的 finish_baseline_round **没有把 c_k 写到 FEDVLA_SCAFFOLD_CV_OUT**。

本脚本直接用合成模型跑一遍该路径，定位到底卡在哪一步：
  1) mode 判定（FEDVLA_BASELINE=scaffold）
  2) wglobal 的键空间（必须是「全模型键空间」trunk.trunk.xxx）
  3) c_k 是否真的落盘、键数多少
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "federated"))

import fedvla_baseline_hook as H  # noqa: E402


class FakeModel(torch.nn.Module):
    """参数名形如 trunk.trunk.blocks.*（HPT 的 Policy.trunk 是 ModuleDict ⇒
    模型参数名是 trunk.trunk.xxx，而 aggregate 写出的 trunk.pth 是 trunk.xxx）。"""

    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Module()
        self.trunk.trunk = torch.nn.Module()
        self.trunk.trunk.blocks = torch.nn.Module()
        self.trunk.trunk.blocks.attn = torch.nn.Linear(4, 4)
        self.trunk.trunk.blocks.mlp = torch.nn.Module()
        self.trunk.trunk.blocks.mlp.experts = torch.nn.ModuleList(
            [torch.nn.Linear(4, 8)])
        self.trunk.trunk.blocks.mlp.experts[0].fc1 = torch.nn.Linear(4, 8)


def build_anchor(path: str, model: torch.nn.Module) -> int:
    """模拟 run_fedavg.sh 的 _write_anchor：写「全模型键空间」的裸 dict。"""
    full = {name: p.detach().cpu().clone()
            for name, p in model.named_parameters()}
    torch.save(full, path)
    return len(full)


def scenario(tag: str, anchor_keys: str, wg_round: int = 0) -> bool:
    """anchor_keys='full' 用模型全量键；'nonmoe' 去掉 experts 键（模拟 224 键那种）。"""
    print(f"\n===== 场景: {tag} =====")
    d = tempfile.mkdtemp(prefix="scaffold_cv_")
    model = FakeModel()

    wg_path = os.path.join(d, f"wglobal_r{wg_round}.pt")
    n = build_anchor(wg_path, model)
    if anchor_keys == "nonmoe":
        blob = torch.load(wg_path, map_location="cpu", weights_only=False)
        blob = {k: v for k, v in blob.items() if "experts" not in k}
        torch.save(blob, wg_path)
        n = len(blob)
    print(f"  wglobal 键数 = {n}")

    cv_out = os.path.join(d, "ci_c0_r0.pt")
    os.environ["FEDVLA_BASELINE"] = "scaffold"
    os.environ["FEDVLA_SCAFFOLD_WGLOBAL"] = wg_path
    os.environ["FEDVLA_SCAFFOLD_CV_OUT"] = cv_out
    os.environ["FEDVLA_SCAFFOLD_STEPS"] = "44"
    os.environ["FEDVLA_SCAFFOLD_LR"] = "1e-4"
    os.environ.pop("FEDVLA_SCAFFOLD_C", None)
    os.environ.pop("FEDVLA_SCAFFOLD_CV_LOCAL", None)
    H._FILE_CACHE.clear()

    # 让模型参数带一点梯度，模拟真实训练后
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.randn_like(p) * 1e-3

    try:
        H.finish_baseline_round(model)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] finish_baseline_round 抛异常: {type(e).__name__}: {e}")
        return False

    ok = os.path.exists(cv_out)
    if not ok:
        print("  [FAIL] ci_c0_r0.pt 未落盘 ⇒ SCAFFOLD 必然在下一轮聚合中止")
        return False
    blob = torch.load(cv_out, map_location="cpu", weights_only=False)
    cv = blob.get("cv", blob) if isinstance(blob, dict) else blob
    print(f"  [OK] 已落盘 ci_c0_r0.pt  键数={len(cv)}")
    if len(cv) == 0:
        print("  [FAIL] 落盘但键数为 0 ⇒ 控制变量是空的，等于 no-op")
        return False
    return True


def main() -> int:
    r1 = scenario("全模型键空间（trunk.trunk.*，含 experts）", "full")
    r2 = scenario("去掉 experts 的键（模拟 224 键那种）", "nonmoe")
    print("\n===== 结论 =====")
    print(f"  全模型键空间落盘: {'通过' if r1 else '失败'}")
    print(f"  无 experts 键落盘: {'通过' if r2 else '失败'}")
    return 0 if (r1 and r2) else 1


if __name__ == "__main__":
    sys.exit(main())
