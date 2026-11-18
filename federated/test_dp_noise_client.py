#!/usr/bin/env python3
"""dp_noise_client.py 的离线单元测试（不需要 GPU / 真实模型）。

验证 4 件事：
  1) ‖Δ‖₂ 的测量正确（与手算一致）；
  2) 裁剪比例 scale = min(1, C/‖Δ‖₂) 正确；
  3) 加噪后噪声的 L2 范数 ≈ σ·C（这是 DP 语义的关键：噪声倍率 = σ）；
  4) stems./heads.（个性化参数）**不被加噪** —— 它们不上行。

跑法：python test_dp_noise_client.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "dp_noise_client.py")


def make_state(seed: int, n_shared: int = 1000):
    g = torch.Generator().manual_seed(seed)
    sd = {
        "trunk.trunk.layer0.w": torch.randn(n_shared, generator=g),
        "normalizer.task1.mean": torch.randn(n_shared, generator=g),
        "modalities_tokens.vision": torch.randn(n_shared, generator=g),
        "stems.task1.stem": torch.randn(n_shared, generator=g),   # 个性化：不应加噪
        "heads.task1.head": torch.randn(n_shared, generator=g),    # 个性化：不应加噪
    }
    return sd


def run(args):
    r = subprocess.run([sys.executable, SCRIPT] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"[!!] 子进程失败 rc={r.returncode}")
    return r.stdout


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="dpnoise_")
    try:
        cdir = os.path.join(tmp, "client")
        gdir = os.path.join(tmp, "global")
        os.makedirs(cdir)
        os.makedirs(gdir)

        prev = make_state(1)
        cur = make_state(2)
        # 人为让上行部分的偏移范数 = 10（便于手算）
        for k in ("trunk.trunk.layer0.w", "normalizer.task1.mean",
                  "modalities_tokens.vision"):
            cur[k] = prev[k] + torch.full_like(prev[k], 1.0)
        # 上行 3 组各 1000 维，偏移恒为 1.0 ⇒ Δ 的 L2 = sqrt(3000)
        expect_l2 = (3000.0) ** 0.5

        torch.save(prev, os.path.join(gdir, "model.pth"))
        torch.save(cur, os.path.join(cdir, "model.pth"))

        # ---- 1) 范数测量 ----
        statf = os.path.join(tmp, "stat.jsonl")
        out = run(["--client-dir", cdir, "--prev-global", gdir,
                   "--calibrate", "--stat-out", statf])
        rec = json.loads(open(statf, encoding="utf-8").readline())
        got = float(rec["delta_l2"])
        print(f"[1] ‖Δ‖₂ 测量 = {got:.6f}，期望 = {expect_l2:.6f}")
        assert abs(got - expect_l2) < 1e-3, "范数测量不对"

        # ---- 2) 裁剪 + 加噪 ----
        # ⚠️ DP-SGD 的标准语义：噪声是 **per-coordinate** N(0, (σ·C)²)，
        #    所以总噪声 L2 = σ·C·√d（d = 该组坐标数），**不是** σ·C。
        #    ⇒ "噪声/信号比" = σ·C·√d / C = σ·√d。
        #    d=17.1M（FedVLA 上行）时 √d≈4134 ⇒ σ=1e-4 才是噪声与信号同量级的点。
        #    这与隐私侧 dp.py 的 "σ·C·√(17.1M)=σ·4139" 是同一件事。
        C = expect_l2 / 2.0        # 故意让 ‖Δ‖ > C ⇒ 触发裁剪
        sigma = 0.1
        d_shared = 3000
        out = run(["--client-dir", cdir, "--prev-global", gdir,
                   "--sigma", str(sigma), "--clip-norm", str(C),
                   "--round", "1", "--client", "0", "--seed", "1234"])
        after = torch.load(os.path.join(cdir, "model.pth"),
                           map_location="cpu", weights_only=False)

        scale = C / expect_l2
        noise_l2 = 0.0
        for k in ("trunk.trunk.layer0.w", "normalizer.task1.mean",
                  "modalities_tokens.vision"):
            d = after[k].float() - prev[k].float()
            noise = d - torch.full_like(d, scale)      # 减去裁剪后的确定性部分
            noise_l2 += float(torch.sum(noise * noise))
        noise_l2 = float(noise_l2) ** 0.5
        target = sigma * C * math.sqrt(d_shared)       # per-coordinate 噪声 ⇒ ×√d
        print(f"[2] 噪声 L2 = {noise_l2:.4f}，目标 σ·C·√d = {target:.4f}，"
              f"相对偏差 = {abs(noise_l2 - target) / target:.2%}")
        assert abs(noise_l2 - target) / target < 0.05, "噪声范数与 σ·C·√d 不符"

        # 逐坐标噪声标准差应 ≈ σ·C
        per_coord_std = float(torch.std(
            (after["trunk.trunk.layer0.w"].float() - prev["trunk.trunk.layer0.w"].float()
             - torch.full_like(prev["trunk.trunk.layer0.w"].float(), scale)).flatten()))
        print(f"[2b] 逐坐标噪声 std = {per_coord_std:.4f}，期望 σ·C = {sigma * C:.4f}")
        assert abs(per_coord_std - sigma * C) / (sigma * C) < 0.05, "逐坐标 std 不符"

        # ---- 3) 个性化参数未被加噪 ----
        for k in ("stems.task1.stem", "heads.task1.head"):
            same = torch.equal(after[k], cur[k])
            print(f"[3] {k} 未被改动 = {same}")
            assert same, "个性化参数被加噪了（物理上不成立）"

        # ---- 4) 上行部分确实变了 ----
        changed = not torch.equal(after["trunk.trunk.layer0.w"],
                                  cur["trunk.trunk.layer0.w"])
        print(f"[4] 上行参数已被加噪改写 = {changed}")
        assert changed, "上行参数没有被加噪"

        # ---- 5) σ=0 时不应有任何改动 ----
        torch.save(cur, os.path.join(cdir, "model.pth"))
        run(["--client-dir", cdir, "--prev-global", gdir,
             "--sigma", "0", "--clip-norm", "1.0", "--round", "1", "--client", "0"])
        z = torch.load(os.path.join(cdir, "model.pth"),
                       map_location="cpu", weights_only=False)
        untouched = all(torch.equal(z[k], cur[k]) for k in cur)
        print(f"[5] σ=0 时所有参数保持原样 = {untouched}")
        assert untouched, "σ=0 却改动了文件"

        print("\n[TEST PASS] dp_noise_client.py 全部 5 项校验通过")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
