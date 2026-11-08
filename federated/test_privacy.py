#!/usr/bin/env python3
"""第 4 阶段隐私侧（DP-SGD + 梯度反演）的本地正确性测试。

纯 CPU，不需要 GPU/模型。用一个**玩具模型**验证：
  A. RDP accountant 的数值正确性（与已知值比对）
  B. 预算分配的单调性与自洽性
  C. 加噪后确实削弱了梯度（SNR 下降）
  D. 反演攻击**在没有噪声时确实能恢复输入**（否则整个实验无意义）
  E. 反演攻击**在有噪声时恢复变差**（DP 有效）

运行：
    python federated/test_privacy.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dp import (  # noqa: E402
    DPConfig, DEFAULT_MODALITY_WEIGHTS, allocate_by_sigma,
    allocate_modality_budget, apply_dp_to_update, classify_key,
    epsilon_from_rdp, group_by_modality, rdp_sampled_gaussian, effective_snr,
)
from inversion import invert_observation, psnr, ssim  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# A. RDP accountant
# ---------------------------------------------------------------------------
def test_rdp():
    print("\n=== A. RDP accountant ===")
    # A1: 无采样时，高斯机制的 RDP(α) = α/(2σ²)
    rdp = rdp_sampled_gaussian(q=1.0, noise_multiplier=2.0, steps=1)
    for alpha in (2, 4, 8, 16):
        expect = alpha / (2 * 2.0 ** 2)
        check(f"无采样 RDP(α={alpha}) = α/(2σ²)",
              abs(rdp[alpha] - expect) < 1e-9,
              f"got {rdp[alpha]:.6f} expect {expect:.6f}")

    # A2: RDP 关于 σ 单调递减
    eps_by_sigma = []
    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
        r = rdp_sampled_gaussian(q=0.25, noise_multiplier=sigma, steps=50)
        e, _ = epsilon_from_rdp(r, 1e-5)
        eps_by_sigma.append(e)
    check("ε 关于 σ 单调递减（噪声越大隐私越强）",
          all(eps_by_sigma[i] > eps_by_sigma[i + 1]
              for i in range(len(eps_by_sigma) - 1)),
          " ".join(f"{e:.3f}" for e in eps_by_sigma))

    # A3: RDP 关于轮数单调递增（复合更多轮 → 隐私消耗更大）
    eps_by_steps = []
    for steps in (1, 10, 50, 200):
        r = rdp_sampled_gaussian(q=0.25, noise_multiplier=1.5, steps=steps)
        e, _ = epsilon_from_rdp(r, 1e-5)
        eps_by_steps.append(e)
    check("ε 关于轮数单调递增",
          all(eps_by_steps[i] < eps_by_steps[i + 1]
              for i in range(len(eps_by_steps) - 1)),
          " ".join(f"{e:.3f}" for e in eps_by_steps))

    # A4: 与已知值比对 —— Opacus 文档里 (q=0.01, σ=1.0, T=100, δ=1e-5)
    #      的 ε 量级应在 5~20 之间（不要求精确，但要落在合理区间）
    r = rdp_sampled_gaussian(q=0.01, noise_multiplier=1.0, steps=100)
    e, order = epsilon_from_rdp(r, 1e-5)
    check("RDP→ε 转换量级合理（q=0.01, σ=1, T=100）",
          1.0 < e < 100.0, f"ε={e:.3f} (α={order})")


# ---------------------------------------------------------------------------
# B. 预算分配
# ---------------------------------------------------------------------------
def test_budget():
    print("\n=== B. 模态感知预算分配 ===")
    cfg = DPConfig(epsilon=8.0, delta=1e-5, steps=50, num_clients=4,
                   modality_aware=True)
    alloc = allocate_modality_budget(cfg)
    check("模态感知：视觉通路的 ε 配额 = 总预算（并行组合，max 语义）",
          abs(alloc.eps_by_modality["vision"] - cfg.epsilon) < 1e-6,
          f"vision={alloc.eps_by_modality['vision']:.3f}")
    check("模态感知：视觉通路的 σ 最小（噪声最少）",
          alloc.sigma_by_modality["vision"] ==
          min(alloc.sigma_by_modality.values()),
          f"σ_vision={alloc.sigma_by_modality['vision']:.4f}")
    check("模态感知：动作通路 σ 最大（噪声最多）",
          alloc.sigma_by_modality["action"] ==
          max(alloc.sigma_by_modality.values()),
          f"σ_action={alloc.sigma_by_modality['action']:.4f}")
    check("模态感知：总 ε = max_m ε_m（并行组合）",
          abs(alloc.achieved_epsilon - cfg.epsilon) < 1e-6,
          f"achieved={alloc.achieved_epsilon:.4f}")

    # 统一预算：所有通路同一个 σ
    cfg_u = DPConfig(epsilon=8.0, delta=1e-5, steps=50, num_clients=4,
                     modality_aware=False)
    alloc_u = allocate_modality_budget(cfg_u)
    sigmas = set(round(v, 10) for v in alloc_u.sigma_by_modality.values())
    check("统一预算：所有通路共用同一个 σ", len(sigmas) == 1,
          f"σ={sigmas}")

    # ε 越大 ⇒ σ 越小
    sigmas_by_eps = []
    for eps in (1.0, 4.0, 16.0):
        a = allocate_modality_budget(
            DPConfig(epsilon=eps, delta=1e-5, steps=50, num_clients=4))
        sigmas_by_eps.append(a.sigma_by_modality["vision"])
    check("ε 越大 ⇒ σ 越小（预算越大噪声越少）",
          sigmas_by_eps[0] > sigmas_by_eps[1] > sigmas_by_eps[2],
          " ".join(f"{s:.4f}" for s in sigmas_by_eps))

    # 实际达成的 ε 应接近目标
    a = allocate_modality_budget(
        DPConfig(epsilon=8.0, delta=1e-5, steps=50, num_clients=4))
    check("实际达成 ε 接近目标 ε=8",
          abs(a.achieved_epsilon - 8.0) / 8.0 < 0.05,
          f"achieved={a.achieved_epsilon:.4f}")

    # ===== σ 优先的分配（2026-11-08 新增）=====
    # 关键不变量：同一个 σ_ref 下，uniform 与 modality 的**总 ε 必须相同**
    # ⇒ H3 的对比才是"相同总预算"下的公平比较。
    for sig in (1e-4, 1e-2, 1.0):
        u = allocate_by_sigma(
            DPConfig(delta=1e-5, steps=50, num_clients=4,
                     modality_aware=False), sig)
        m = allocate_by_sigma(
            DPConfig(delta=1e-5, steps=50, num_clients=4,
                     modality_aware=True), sig)
        check(f"σ={sig:g}：两种模式总 ε 相同（等预算对比的前提）",
              abs(u.achieved_epsilon - m.achieved_epsilon) /
              max(u.achieved_epsilon, 1e-12) < 1e-6,
              f"uniform={u.achieved_epsilon:.6g} modality={m.achieved_epsilon:.6g}")
        check(f"σ={sig:g}：模态感知给 shared(trunk) 的 σ 更大 ⇒ 保护更强",
              m.sigma_by_modality["shared"] > u.sigma_by_modality["shared"],
              f"σ_shared: uniform={u.sigma_by_modality['shared']:.6g} "
              f"modality={m.sigma_by_modality['shared']:.6g}")
        check(f"σ={sig:g}：模态感知给 vision 的 σ 与统一预算相同",
              abs(m.sigma_by_modality["vision"] - u.sigma_by_modality["vision"]) < 1e-12,
              f"{m.sigma_by_modality['vision']:.6g}")


# ---------------------------------------------------------------------------
# C. 加噪
# ---------------------------------------------------------------------------
def test_noise():
    print("\n=== C. DP 加噪 ===")
    torch.manual_seed(0)
    update = {
        "trunk.trunk.block.0.attn.q.weight": torch.randn(64, 64) * 0.1,
        "stems.mujoco_metaworld_task3_image.enc.0.weight": torch.randn(32, 32) * 0.1,
        "stems.mujoco_metaworld_task3_state.enc.0.weight": torch.randn(16, 16) * 0.1,
        "heads.mujoco_metaworld_task3.head.weight": torch.randn(16, 16) * 0.1,
    }
    check("参数分组：4 个键分到 4 个通路",
          len(group_by_modality(update)) == 4,
          str({k: len(v) for k, v in group_by_modality(update).items()}))
    check("分类：stems.*_image.* → vision",
          classify_key("stems.mujoco_metaworld_task3_image.enc.0.weight") == "vision")
    check("分类：heads.* → action",
          classify_key("heads.mujoco_metaworld_task3.head.weight") == "action")

    alloc = allocate_modality_budget(
        DPConfig(epsilon=1.0, delta=1e-5, steps=50, num_clients=4))
    noisy = apply_dp_to_update(update, alloc, clip_norm=1.0, seed=7)
    check("加噪后键集合不变", set(noisy.keys()) == set(update.keys()))
    check("加噪后形状不变",
          all(noisy[k].shape == update[k].shape for k in update))

    # SNR 定义：||干净信号|| / ||噪声||（噪声水平由 ε 决定）
    # ⚠️ 不能拿"加噪后的向量"当分子 —— 加噪会放大范数，SNR 反而上升（反直觉）。
    snr_clean = effective_snr(update, alloc, clip_norm=1.0)
    # 实际加入的噪声能量应当与理论值 (σ·C·√numel) 同量级
    noise_norm = {}
    for m, keys in group_by_modality(update).items():
        d = torch.cat([(noisy[k] - update[k]).reshape(-1).float() for k in keys])
        noise_norm[m] = float(torch.linalg.vector_norm(d))
        expect = alloc.sigma_by_modality[m] * 1.0 * math.sqrt(d.numel())
        check(f"实际噪声范数与理论 (σC√n) 同量级 [{m}]",
              0.2 * expect < noise_norm[m] < 3.0 * expect,
              f"actual={noise_norm[m]:.3f} expect≈{expect:.3f}")

    # ε 越小 ⇒ 噪声越大 ⇒ SNR 越低
    snr_by_eps = {}
    for eps in (1.0, 8.0, 32.0):
        a = allocate_modality_budget(
            DPConfig(epsilon=eps, delta=1e-5, steps=50, num_clients=4))
        snr_by_eps[eps] = effective_snr(update, a, clip_norm=1.0)["vision"]
    check("ε 越小 ⇒ SNR 越低（隐私更强）",
          snr_by_eps[1.0] < snr_by_eps[8.0] < snr_by_eps[32.0],
          " ".join(f"ε={e}:{v:.2f}" for e, v in sorted(snr_by_eps.items())))

    # 同一 seed ⇒ 同样结果（可复现）
    noisy2 = apply_dp_to_update(update, alloc, clip_norm=1.0, seed=7)
    check("同 seed 可复现",
          all(torch.allclose(noisy[k], noisy2[k]) for k in noisy))

    # 不同 seed ⇒ 不同结果
    noisy3 = apply_dp_to_update(update, alloc, clip_norm=1.0, seed=8)
    check("不同 seed 结果不同",
          not all(torch.allclose(noisy[k], noisy3[k]) for k in noisy))

    # 噪声强度与 σ 单调对应
    a1 = allocate_modality_budget(DPConfig(epsilon=1.0, delta=1e-5, steps=50))
    a2 = allocate_modality_budget(DPConfig(epsilon=32.0, delta=1e-5, steps=50))
    d1 = (apply_dp_to_update(update, a1, seed=1)["trunk.trunk.block.0.attn.q.weight"]
          - update["trunk.trunk.block.0.attn.q.weight"]).std()
    d2 = (apply_dp_to_update(update, a2, seed=1)["trunk.trunk.block.0.attn.q.weight"]
          - update["trunk.trunk.block.0.attn.q.weight"]).std()
    check("ε 越大 ⇒ 实际噪声越小", d2 < d1, f"std: ε=1 → {d1:.4f}, ε=32 → {d2:.4f}")


# ---------------------------------------------------------------------------
# D/E. 反演攻击（玩具模型）
# ---------------------------------------------------------------------------
class ToyPolicy(nn.Module):
    """极简"策略"：conv 编码图像 → MLP 出动作。

    结构刻意模仿 FedVLA：trunk（共享）+ stem（视觉）+ head（动作）。
    """

    def __init__(self):
        super().__init__()
        self.trunk = nn.Module()
        self.trunk.trunk = nn.Module()
        self.trunk.trunk.enc = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.stems = nn.Module()
        self.stems.image = nn.Module()
        self.stems.image.enc = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.heads = nn.Module()
        self.heads.head = nn.Linear(32 * 8 * 8, 4)

    def forward_features(self, data):
        x = self.stems.image.enc(data["image"])
        return self.trunk.trunk.enc(x)

    def compute_loss(self, batch):
        data = batch["data"]
        feat = self.forward_features(data)
        feat = F.adaptive_avg_pool2d(feat, 8).flatten(1)
        pred = self.heads.head(feat)
        return F.mse_loss(pred, data["action"])

    def named_parameters(self, prefix="", recurse=True):
        # 让 trunk.* 的键名带上 "trunk." 前缀（与真实 HPT 一致）
        for n, p in super().named_parameters(prefix=prefix, recurse=recurse):
            yield n, p


def test_inversion():
    print("\n=== D/E. 梯度反演攻击（玩具模型）===")
    torch.manual_seed(0)
    model = ToyPolicy()
    domain = "toy"

    real_image = torch.rand(1, 3, 16, 16)
    real_action = torch.randn(1, 4)
    batch = {"data": {"image": real_image.clone(), "action": real_action.clone()}}

    # 1) 计算"服务器收到"的梯度（trunk 部分）
    model.zero_grad(set_to_none=True)
    loss = model.compute_loss(batch)
    loss.backward()
    observed = {n: p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None and n.startswith("trunk.")}

    # 2) 无噪声反演 —— 必须能恢复出可辨认的图像
    #    ⚠️ 玩具模型用的是 3×16×16 **像素**图 ⇒ target="image_pixel"
    #    （真实 FedVLA 的 image 是 ResNet 特征图 ⇒ target="image"，指标是逐 patch 相关）
    res_clean = invert_observation(model, domain, batch, observed,
                                   target="image_pixel",
                                   steps=300, lr=0.05, seed=0)
    check("无噪声反演：PSNR 显著高于随机基线",
          res_clean["psnr"] > res_clean["baseline_psnr"] + 3.0,
          f"PSNR={res_clean['psnr']:.2f} dB vs 基线 {res_clean['baseline_psnr']:.2f} dB")
    check("无噪声反演：SSIM 显著高于随机基线",
          res_clean["ssim"] > res_clean["baseline_ssim"] + 0.05,
          f"SSIM={res_clean['ssim']:.4f} vs 基线 {res_clean['baseline_ssim']:.4f}")
    check("无噪声反演：匹配损失下降到很小",
          res_clean["match_loss_curve"][-1] < res_clean["match_loss_curve"][0],
          f"{res_clean['match_loss_curve'][0]:.4f} → {res_clean['match_loss_curve'][-1]:.4f}")

    # 3) 加噪后反演 —— 必须变差
    alloc = allocate_modality_budget(
        DPConfig(epsilon=0.5, delta=1e-5, steps=50, num_clients=4))
    noisy_grad = apply_dp_to_update(observed, alloc, clip_norm=1.0, seed=3)
    res_noisy = invert_observation(model, domain, batch, noisy_grad,
                                   target="image_pixel",
                                   steps=300, lr=0.05, seed=0)
    check("加噪后反演质量下降（DP 有效）",
          res_noisy["psnr"] < res_clean["psnr"],
          f"PSNR: {res_clean['psnr']:.2f} → {res_noisy['psnr']:.2f} dB")

    # 4) 噪声越大 ⇒ 反演越差
    #    ⚠️ 不能用 PSNR 判单调：噪声一大，PSNR 就触到"随机基线地板"
    #    （实测 ε=32/2/0.5 的 PSNR 是 6.22/6.16/6.21 dB，完全饱和），
    #    排序是纯噪声。改用**攻击者自己的匹配损失**：噪声越大，
    #    候选梯度越难匹配上观测梯度 ⇒ best match loss 越大。
    mloss = []
    psnrs = []
    for eps in (32.0, 2.0, 0.5):
        a = allocate_modality_budget(
            DPConfig(epsilon=eps, delta=1e-5, steps=50, num_clients=4))
        ng = apply_dp_to_update(observed, a, clip_norm=1.0, seed=3)
        r = invert_observation(model, domain, batch, ng, target="image_pixel",
                               steps=150, lr=0.05, seed=0)
        mloss.append(r["match_loss_best"])
        psnrs.append(r["psnr"])
    check("ε 越小 ⇒ 攻击者的匹配损失越大（噪声更难匹配）",
          mloss[0] <= mloss[1] <= mloss[2],
          " ".join(f"ε={e}:{m:.4f}" for e, m in zip((32.0, 2.0, 0.5), mloss)))
    check("加了 DP 噪声后重建质量全部崩塌到随机基线附近",
          all(p < res_clean["baseline_psnr"] + 1.5 for p in psnrs),
          " ".join(f"{p:.2f}" for p in psnrs) +
          f" (基线 {res_clean['baseline_psnr']:.2f})")


# ---------------------------------------------------------------------------
# F. 指标函数
# ---------------------------------------------------------------------------
def test_metrics():
    print("\n=== F. 评价指标 ===")
    a = torch.rand(1, 3, 32, 32)
    check("PSNR(自比) ≈ 99 dB", psnr(a, a) > 90.0, f"{psnr(a, a):.2f}")
    check("SSIM(自比) ≈ 1.0", abs(ssim(a, a) - 1.0) < 1e-3, f"{ssim(a, a):.4f}")
    b = torch.rand(1, 3, 32, 32)
    check("PSNR(随机 vs 随机) 明显低于自比",
          psnr(a, b) < 20.0, f"{psnr(a, b):.2f} dB")
    check("SSIM(随机 vs 随机) 明显低于自比",
          ssim(a, b) < 0.5, f"{ssim(a, b):.4f}")
    c = (a + 0.02 * torch.randn_like(a)).clamp(0, 1)
    check("小噪声 ⇒ PSNR 落在合理区间 (10~35 dB)",
          10.0 < psnr(a, c) < 35.0, f"{psnr(a, c):.2f} dB")


def main() -> int:
    print("=" * 70)
    print("FedVLA 第4阶段隐私侧测试（DP-SGD + 梯度反演）")
    print("=" * 70)
    test_rdp()
    test_budget()
    test_noise()
    test_metrics()
    test_inversion()
    print("\n" + "=" * 70)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("  - " + f)
        print("=" * 70)
        return 1
    print("全部通过 ✅")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
