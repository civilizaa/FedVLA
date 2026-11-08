#!/usr/bin/env python3
"""FedVLA 第 4 阶段隐私侧：梯度反演攻击（DLG）。

## 攻击设定（论文 6.3 节）

- 威胁模型：**诚实但好奇**的服务器遵守协议，但会分析收到的上行更新。
- 服务器拿到的是**被 DP 噪声污染后的**上行更新 Δθ（FedVLA 的上行 =
  `trunk.*` + `normalizer.<domain>` + `modalities_tokens.*`；stems/heads 个性化，不上行）。
- 攻击目标：重建客户端的私有观测与动作。

## 方法

经典 **DLG（Deep Leakage from Gradients）**：给定观察到的梯度 g，
优化候选潜变量使在候选上算出的梯度尽量接近 g：

    min_{x̂}  ‖ ∇_θ L(f_θ(x̂), ŷ) − g ‖²

在 VLA 场景下，"标签"对应**动作序列**。

## ⚠️ 三个与 FedVLA 架构强相关的事实（2026-11-08 实测）

### 1. 反演目标是**特征图**而非像素
Meta-World 的 zarr 里 `image` 存的是 **ResNet 提取的 512 维特征**
（形状 `(N, 1, 49, 512)`，取值范围 [0, 15.77]），不是 512×512 像素。
所以：
- 反演目标 x̂ 的形状 = `(1, 1, 49, 512)`；
- **不能用 sigmoid 把 x̂ 约束到 [0,1]**（那是像素的做法），会直接毁掉优化；
- 指标改成**特征域**的逐 patch Pearson 相关 + 相对 L2，PSNR 只作辅助。

### 2. 必须取梯度键的**交集**
MoE 的 top-k 路由是**输入相关**的：不同输入激活不同专家 ⇒ 候选前向的梯度
张量集合是服务器收到的梯度集合的**真子集**（实测 15.82M vs 17.13M）。
服务器只收到"被激活专家"的梯度 ⇒ 攻击者只能用交集匹配，否则维度对不上。
默认 `match_mode="dense"`（只匹配 blocks 0-7 的稠密参数），
既避开路由不一致，又把 double-backward 的开销降到 1/3。

### 3. 语言通路在本配置下不存在
实测 `policy.modalities = ['image', 'state']`，`loss(有语言) − loss(无语言) = 0`。
HPT 的 T5 语言嵌入没有被消费 ⇒ 模态敏感度分析覆盖 **视觉/状态/动作** 三通路。

## 用法

```python
from inversion import invert_observation

res = invert_observation(
    model=policy, domain="mujoco_metaworld_task3",
    real_batch={"image": img, "state": st, "action": ac},
    observed_grad=noisy_trunk_grad,      # 服务器收到的（已加噪的）上行更新
    target="image",                      # "image" | "state" | "action"
    steps=200, lr=0.05,
)
res["corr_mean"], res["rel_l2"], res["baseline_corr_mean"]
```
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ⚠️ DLG 的硬前提：注意力必须走**数学后端**。
#    PyTorch 默认用 `aten::_scaled_dot_product_efficient_attention`，
#    它的 backward **没有二阶导实现** ⇒ DLG 的 create_graph=True 会报
#    "derivative for aten::_scaled_dot_product_efficient_attention_backward
#     is not implemented"。关掉 flash / mem-efficient，强制 math。
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
    """峰值信噪比（dB）。"""
    mse = float(F.mse_loss(a.float(), b.float()))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> float:
    """结构相似度（单通道版，11×11 高斯窗）。"""
    if a.dim() == 3:
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
    if a.dim() != 4:
        raise ValueError("ssim 需要 3D 或 4D 输入")
    coords = torch.arange(11, dtype=a.dtype, device=a.device) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window2d = torch.outer(g, g)
    window = window2d.expand(a.shape[1], 1, 11, 11).contiguous()
    pad = 5
    mu1 = F.conv2d(a, window, padding=pad, groups=a.shape[1])
    mu2 = F.conv2d(b, window, padding=pad, groups=b.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(a * a, window, padding=pad, groups=a.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=pad, groups=b.shape[1]) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=pad, groups=b.shape[1]) - mu1_mu2
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
        ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(s.mean())


def patch_correlation(x_hat: torch.Tensor, x_true: torch.Tensor) -> Tuple[float, float]:
    """逐 patch Pearson 相关（沿最后一维），返回 (mean, max)。

    x_hat / x_true 形状 (..., P, D)：P=49 个 patch，D=512 维特征。
    """
    a = x_hat.reshape(-1, x_hat.shape[-1]).float()
    b = x_true.reshape(-1, x_true.shape[-1]).float()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    c = num / den
    return float(c.mean()), float(c.max())


def rel_l2(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """相对 L2 误差 ‖x̂−x‖ / ‖x‖。"""
    return float(torch.linalg.vector_norm(x_hat.float() - x_true.float()) /
                 (torch.linalg.vector_norm(x_true.float()) + 1e-12))


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


# ---------------------------------------------------------------------------
# 梯度工具
# ---------------------------------------------------------------------------


def _flat_grad(grads: Dict[str, torch.Tensor], keys: Sequence[str]
               ) -> Optional[torch.Tensor]:
    parts = []
    for k in keys:
        g = grads.get(k)
        if g is None:
            continue
        parts.append(g.reshape(-1).float())
    if not parts:
        return None
    return torch.cat(parts)


def trunk_keys(model: torch.nn.Module) -> List[str]:
    """模型里属于共享 trunk 的参数键。

    ⚠️ Policy.trunk 是 nn.ModuleDict ⇒ policy 内的完整键是 `trunk.trunk.xxx`，
    前缀 `trunk.` 依然匹配（实测门0：264 键）。
    """
    return [n for n, _ in model.named_parameters() if n.startswith("trunk.")]


def dense_trunk_keys(model: torch.nn.Module) -> List[str]:
    """只取**稠密**（非 MoE）block 的参数键：blocks.0 ~ blocks.7。

    为什么要单独提供：MoE 的 top-k 路由随输入变化 ⇒ 候选前向与服务器
    收到的梯度在 MoE 块上键集合不一致，且 double-backward 成本高 3 倍。
    稠密块 0-7 与输入无关的假设更强、匹配更稳定。
    """
    out = []
    for n, _ in model.named_parameters():
        if not n.startswith("trunk."):
            continue
        if ".blocks." in n:
            try:
                bi = int(n.split(".blocks.")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if bi >= 8:
                continue
        out.append(n)
    return out


def compute_loss_for_batch(model: torch.nn.Module, domain: str,
                           batch: Dict, device: torch.device) -> torch.Tensor:
    """在给定 batch 上算 policy 的 loss（与训练用同一个 compute_loss）。"""
    data = {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}
    # 训练时 normalizer 会就地改写 data["action"]/data["state"]，先复制避免污染
    data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    return model.compute_loss({"domain": [domain], "data": data})


def _candidate_grads(model: torch.nn.Module, domain: str, data: Dict,
                     match_keys: Sequence[str], latent_key: str,
                     device: torch.device):
    """在候选潜变量上算 trunk 梯度（保留计算图，供二阶反传）。"""
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    d[latent_key] = d.pop("__LATENT__")
    loss = model.compute_loss({"domain": [domain], "data": d})
    keyset = set(match_keys)
    names = [n for n, _ in model.named_parameters() if n in keyset]
    params = [q for n, q in model.named_parameters() if n in keyset]
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True,
                                allow_unused=True)
    return loss, {n: g for n, g in zip(names, grads)}


# ---------------------------------------------------------------------------
# 反演攻击
# ---------------------------------------------------------------------------

#: 三条通路的反演目标定义
#:   key        —— 候选潜变量替换 batch 里的哪个键
#:   param      —— 参数化："none"（原值，特征域用）| "sigmoid"（约束到 [0,1]，像素域用）
#:   metric     —— "patch"（逐 patch 相关）| "pixel"（PSNR/SSIM）| "vector"（余弦）
TARGET_SPECS = {
    "image": {"key": "image", "param": "none", "metric": "patch"},
    "image_pixel": {"key": "image", "param": "sigmoid", "metric": "pixel"},
    "state": {"key": "state", "param": "none", "metric": "vector"},
    "action": {"key": "action", "param": "none", "metric": "vector"},
}


def invert_observation(
    model: torch.nn.Module,
    domain: str,
    real_batch: Dict,
    observed_grad: Dict[str, torch.Tensor],
    target: str = "image",
    steps: int = 200,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
    match_keys: Optional[Sequence[str]] = None,
    match_mode: str = "dense",
    seed: int = 0,
    verbose: bool = False,
    task_loss_weight: float = 0.05,
) -> Dict:
    """从观察到的（可能已加噪的）上行更新反演私有观测 / 动作。

    参数
    ----
    model         —— policy（含 trunk + stem + head）
    domain        —— 域名，如 "mujoco_metaworld_task3"
    real_batch    —— 真实 batch：{"image":…, "state":…, "action":…}
    observed_grad —— 服务器收到的上行更新（参数名 → 梯度张量）
    target        —— "image" | "state" | "action"
    match_keys    —— 用于匹配的参数键；默认按 match_mode 取
    match_mode    —— "dense"（只 blocks0-7，推荐）| "all"（全部 trunk）

    返回
    ----
    dict：match_loss 曲线、recon（numpy）、各项指标、随机基线指标
    """
    if target not in TARGET_SPECS:
        raise ValueError(f"target 必须是 {list(TARGET_SPECS)} 之一")
    device = device or next(model.parameters()).device
    spec = TARGET_SPECS[target]
    latent_key = spec["key"]

    if isinstance(real_batch.get("data"), dict):      # 兼容旧调用格式
        real_batch = real_batch["data"]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # ---- 准备真实潜变量与固定输入 ----
    true_t = real_batch[latent_key].detach().to(device).float()
    fixed = {k: v.detach().to(device).float()
             for k, v in real_batch.items() if k != latent_key}

    if match_keys is None:
        match_keys = (dense_trunk_keys(model) if match_mode == "dense"
                      else trunk_keys(model))
    match_keys = [k for k in match_keys if observed_grad.get(k) is not None]
    if not match_keys:
        raise ValueError("observed_grad 里没有可匹配的参数")

    tgt_map = {k: observed_grad[k].reshape(-1).float() for k in match_keys
               if observed_grad.get(k) is not None}

    # ---- 初始化候选 ----
    # ⚠️ 默认用**随机初始化**（诚实攻击者：他只有梯度，不知道这条样本的内容）。
    #    绝不能用"真值 + 小噪声"当主结果 —— 那等于把原图直接还给攻击者，
    #    重建指标会漂亮但毫无意义（2026-11-08 踩过：corr=0.9944 全是初始化的功劳）。
    #    `FEDVLA_INV_INIT=prior` 仅作"先验上界"的对照实验。
    #    另外：特征域目标不能用 sigmoid（会把值域压到 [0,1]，
    #    而 ResNet 特征的取值范围是 [0, 15.77]）。
    prior_init = os.environ.get("FEDVLA_INV_INIT", "random")
    if prior_init == "prior":
        base = true_t.clone()
        noise = torch.randn(true_t.shape, generator=gen).to(device)
        latent = (base + 0.05 * noise.float() * (base.std() + 1e-6)).requires_grad_(True)
    else:
        # 随机初始化：用与真值同量纲的高斯噪声（尺度对齐，避免优化初期梯度消失）
        latent = (torch.randn(true_t.shape, generator=gen).to(device).float()
                  * (true_t.std() + 1e-6)).requires_grad_(True)

    def materialize(x: torch.Tensor) -> torch.Tensor:
        """把潜变量参数化到模型的输入空间。"""
        return torch.sigmoid(x) if spec["param"] == "sigmoid" else x

    opt = torch.optim.Adam([latent], lr=lr)

    # ---- 模型设置：只放开被匹配的参数，其余冻结 ----
    model.train(False)
    model.eval()
    for n_, q in model.named_parameters():
        q.requires_grad_(n_ in set(match_keys))

    losses: List[float] = []
    best = {"loss": float("inf"), "recon": None}
    t0 = time.time()
    n_common = 0

    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        data = dict(fixed)
        data["__LATENT__"] = materialize(latent)
        loss_val, grads = _candidate_grads(model, domain, data, match_keys,
                                            latent_key, device)

        # ⚠️ 取交集：MoE top-k 路由随输入变化，候选梯度键 ⊂ 服务器梯度键
        common = [k for k in match_keys
                  if grads.get(k) is not None and k in tgt_map]
        if not common:
            raise RuntimeError("没有共同的梯度键（候选与观测的梯度集合不相交）")
        cand = torch.cat([grads[k].reshape(-1).float() for k in common])
        tgt = torch.cat([tgt_map[k] for k in common])
        cand_unit = cand / (torch.linalg.vector_norm(cand) + 1e-12)
        tgt_unit = tgt / (torch.linalg.vector_norm(tgt) + 1e-12)

        match = ((cand_unit - tgt_unit) ** 2).sum()
        total = match + task_loss_weight * loss_val
        total.backward()
        opt.step()

        mval = float(match.detach())
        losses.append(mval)
        n_common = len(common)
        cur = materialize(latent).detach()
        if mval < best["loss"]:
            best = {"loss": mval, "recon": cur.clone()}
        if verbose and (step % 25 == 0 or step == steps - 1):
            p_rel = rel_l2(cur, true_t)
            print(f"[inversion:{target}] step {step:4d} match={mval:.6f} "
                  f"task_loss={float(loss_val):.6f} rel_l2={p_rel:.4f} "
                  f"common={len(common)} ({time.time()-t0:.1f}s)", flush=True)

    model.zero_grad(set_to_none=True)
    for n_, q in model.named_parameters():
        q.requires_grad_(False)

    recon = best["recon"]
    result: Dict = {
        "target": target,
        "match_loss_curve": losses,
        "final_match_loss": losses[-1] if losses else None,
        "best_match_loss": best["loss"],
        "recon": recon.cpu().numpy(),
        "real": true_t.cpu().numpy(),
        "n_common_keys": n_common,
        "n_match_keys": len(match_keys),
        "seconds": time.time() - t0,
        "match_mode": match_mode,
    }

    # ---- 指标（按通路口径）----
    # 随机基线 = "什么都没反演出来"的水平。像素域用 [0,1] 均匀噪声，
    # 特征域用同尺度的零均值噪声（与真值同量纲）。
    if spec["metric"] == "pixel":
        rand = torch.rand(true_t.shape, generator=gen).to(device)
        result.update(
            psnr=psnr(recon.cpu(), true_t.cpu(), data_range=1.0),
            ssim=ssim(recon.cpu(), true_t.cpu(), data_range=1.0),
            baseline_psnr=psnr(rand.cpu(), true_t.cpu(), data_range=1.0),
            baseline_ssim=ssim(rand.cpu(), true_t.cpu(), data_range=1.0),
            rel_l2=rel_l2(recon, true_t),
            baseline_rel_l2=rel_l2(rand, true_t),
        )
    elif spec["metric"] == "patch":
        cm, cx = patch_correlation(recon, true_t)
        rand = (torch.randn(true_t.shape, generator=gen).to(device).float()
                * (true_t.std() + 1e-6))
        bm, bx = patch_correlation(rand, true_t)
        result.update(
            corr_mean=cm, corr_max=cx,
            baseline_corr_mean=bm, baseline_corr_max=bx,
            rel_l2=rel_l2(recon, true_t),
            baseline_rel_l2=rel_l2(rand, true_t),
            # 特征量纲 [0, 15.77] 下的 PSNR（仅作辅助指标）
            psnr_feature=psnr(recon.cpu(), true_t.cpu(), data_range=15.77),
        )
    else:   # vector
        cos = cos_sim(recon, true_t)
        base = torch.zeros_like(true_t)
        result.update(
            cos=cos, baseline_cos=cos_sim(base, true_t),
            rel_l2=rel_l2(recon, true_t), baseline_rel_l2=rel_l2(base, true_t),
        )

    # 恢复原状态（调用方可能还要用模型）
    for n_, q in model.named_parameters():
        q.requires_grad_(True)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="梯度反演攻击（干跑用）")
    ap.add_argument("--smoke", action="store_true",
                    help="只跑指标函数的自检（不需要模型）")
    args = ap.parse_args(argv)
    if args.smoke:
        a = torch.rand(1, 3, 64, 64)
        b = (a + 0.05 * torch.randn_like(a)).clamp(0, 1)
        print(f"psnr(self+noise) = {psnr(a, b):.2f} dB")
        print(f"ssim(self+noise) = {ssim(a, b):.4f}")
        c = torch.rand(1, 3, 64, 64)
        print(f"psnr(random)      = {psnr(a, c):.2f} dB")
        print(f"ssim(random)      = {ssim(a, c):.2f} dB")
        x = torch.randn(1, 1, 49, 512)
        y = x + 0.1 * torch.randn_like(x)
        print(f"patch corr(noisy) = {patch_correlation(y, x)}")
        print(f"rel_l2(noisy)      = {rel_l2(y, x):.4f}")
        return 0
    print("请在云端用 run_privacy_scan.sh 调用本模块")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
