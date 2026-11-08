#!/usr/bin/env python3
"""探针 stage3：语言通路判定 + DLG 梯度反演可行性（隐私实验的最大风险点）。

Part A  语言是否被策略消费？（eval 模式、去 dropout ⇒ 确定性）
Part B  干净 trunk 梯度 → DLG 反演「视觉特征图」能不能成功？
Part C  干净 trunk 梯度 → DLG 反演「动作序列」能不能成功？
Part D  每步耗时（用于预算完整扫描）

指标口径（全部在特征/动作域，不是像素域）：
  - 逐 patch Pearson 相关（vision）
  - 相对 L2 误差
  - 余弦相似度（action）
"""
import io
import os
import sys
import time
import traceback

sys.path.insert(0, "/root/HPT")

import numpy as np
import torch

ZARR = "/root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100"
GLOBAL_DIR = "/root/HPT/output/p3_moe_eda_cluster/global_r49"
DOMAIN = "mujoco_metaworld_task3"

OUT = io.StringIO()


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.write(s + "\n")


def hr(t):
    p("\n" + "=" * 70)
    p(t)
    p("=" * 70)


def build_policy():
    import zarr
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from hpt.models.policy import Policy
    from hpt.utils import utils as hutils
    from hpt.utils.normalizer import LinearNormalizer

    cfg = OmegaConf.structured(OmegaConf.load(os.path.join(GLOBAL_DIR, "config.yaml")))
    net = OmegaConf.create(dict(cfg.network))
    net["_target_"] = "hpt.models.policy.Policy"
    pol = instantiate(net)

    z = zarr.open(ZARR, mode="r")
    d = z["data"]

    class DS:
        state_dim = 39
        num_views = 1
        one_action_dim = 4
        action_dim = 32

    hutils.update_network_dim(cfg, DS(), pol)
    pol.init_domain_stem(DOMAIN, cfg.stem)
    nz = LinearNormalizer()
    nz.fit(data={"action": np.asarray(d["action"][:]), "state": np.asarray(d["state"][:])},
           last_n_dims=1, mode="limits")
    pol.init_domain_head(DOMAIN, nz, cfg.head)
    pol.finalize_modules()
    sd = torch.load(os.path.join(GLOBAL_DIR, "model.pth"), map_location="cpu", weights_only=False)
    pol.load_state_dict(sd, strict=False)
    pol.eval()          # 关 dropout ⇒ 前向确定性
    pol.to(torch.device("cuda"))
    return pol, cfg, d


def make_batch(d, idx, device):
    img = torch.from_numpy(np.asarray(d["image"][idx])).float().unsqueeze(0).to(device)
    st = torch.from_numpy(np.asarray(d["state"][idx])).float()[None, None].to(device)
    ac = torch.from_numpy(np.asarray(d["action"][idx:idx + 8])).float().unsqueeze(0).to(device)
    return {"image": img, "state": st, "action": ac}


def flat_grad(grads, keys):
    parts = [grads[k].reshape(-1).float() for k in keys if k in grads and grads[k] is not None]
    return torch.cat(parts) if parts else None


def clean_grad(pol, batch, device):
    """在 eval 模式下算 trunk 梯度（服务器收到的上行更新）。"""
    for n, q in pol.named_parameters():
        q.requires_grad_(n.startswith("trunk."))
    data = {k: v.clone() for k, v in batch.items()}
    loss = pol.compute_loss({"domain": [DOMAIN], "data": data})
    loss.backward()
    g = {n: q.grad.detach().clone() for n, q in pol.named_parameters()
         if n.startswith("trunk.") and q.grad is not None}
    pol.zero_grad(set_to_none=True)
    return g, float(loss)


def dlg(pol, target_grad, match_keys, latent, real, steps=100, lr=0.05,
        device="cuda", extra_fixed=None):
    """通用 DLG：优化 latent（叶子变量）使 trunk 梯度匹配 target_grad。

    latent    —— 候选潜变量张量（已 requires_grad）
    real      —— 真实潜变量（仅用于算指标）
    extra_fixed —— 固定不优化的其它输入（state / action）
    """
    tgt = flat_grad(target_grad, match_keys)
    tgt_unit = (tgt / (torch.linalg.vector_norm(tgt) + 1e-12)).detach()

    opt = torch.optim.Adam([latent], lr=lr)
    t0 = time.time()
    hist = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        data = dict(extra_fixed or {})
        data["__LATENT__"] = latent
        # 拼 batch：latent 取代对应键
        loss_val, grads = _forward_grads(pol, data, match_keys)
        cand = flat_grad(grads, match_keys)
        cand_unit = cand / (torch.linalg.vector_norm(cand) + 1e-12)
        match = ((cand_unit - tgt_unit) ** 2).sum()
        total = match + 0.05 * loss_val
        total.backward()
        opt.step()
        hist.append(float(match.detach()))
        if step % 20 == 0 or step == steps - 1:
            p(f"    step {step:4d} match={float(match):.6f} task_loss={float(loss_val):.6f} "
              f"({time.time()-t0:.1f}s)")
    return hist, time.time() - t0


def _forward_grads(pol, data, match_keys):
    """前向一次，拿到 trunk 梯度（保留计算图用于二阶反传）。"""
    data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    loss = pol.compute_loss({"domain": [DOMAIN], "data": data})
    params = [q for n, q in pol.named_parameters() if n in set(match_keys)]
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True,
                                allow_unused=True)
    return loss, {n: g for n, g in zip([n for n, q in pol.named_parameters()
                                        if n in set(match_keys)], grads)}


def corr_metrics(x_hat, x_true):
    """逐 patch Pearson 相关（沿最后一维）。"""
    a = x_hat.reshape(-1, x_hat.shape[-1]).float()
    b = x_true.reshape(-1, x_true.shape[-1]).float()
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-12
    c = num / den
    rel = float(torch.linalg.vector_norm(x_hat - x_true) /
                (torch.linalg.vector_norm(x_true) + 1e-12))
    return float(c.mean()), float(c.max()), rel


def cos_metrics(a, b):
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    rel = float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-12))
    return cos, rel


def main():
    device = torch.device("cuda")
    pol, cfg, d = build_policy()

    hr("Part A: 模态与语言通路判定")
    p(f"  policy.modalities = {getattr(pol, 'modalities', None)}")
    p(f"  cfg.stem keys      = {list(cfg.stem.keys())}")
    p(f"  stems 模块         = {list(pol.stems.keys())}")
    batch = make_batch(d, 0, device)
    lg = torch.from_numpy(np.asarray(d["language"][0])[:1]).float()[None].to(device)
    with torch.no_grad():
        l1 = float(pol.compute_loss({"domain": [DOMAIN],
                                      "data": {**batch, "image": batch["image"].clone(),
                                               "state": batch["state"].clone(),
                                               "action": batch["action"].clone()}}))
        l2 = float(pol.compute_loss({"domain": [DOMAIN],
                                      "data": {**batch, "image": batch["image"].clone(),
                                               "state": batch["state"].clone(),
                                               "action": batch["action"].clone(),
                                               "language": lg}}))
    p(f"  loss(无语言) = {l1:.8f}")
    p(f"  loss(有语言) = {l2:.8f}")
    p(f"  |Δ| = {abs(l1-l2):.3e}  ⇒ {'语言被消费' if abs(l1-l2) > 1e-6 else '语言【未】进入策略（被忽略）'}")

    hr("Part B/C: 干净梯度 → DLG 可行性")
    g, loss = clean_grad(pol, batch, device)
    p(f"  clean loss={loss:.6f}  trunk 梯度张量数={len(g)}")
    all_keys = list(g.keys())
    p(f"  梯度总维度={sum(v.numel() for v in g.values())}")

    # ---- B: 反演视觉特征图 ----
    p("\n  [B] 目标 = 视觉特征图 (1,1,49,512)")
    x_true = batch["image"].detach()
    x_hat = torch.randn_like(x_true) * 0.1
    x_hat.requires_grad_(True)
    fixed = {"state": batch["state"], "action": batch["action"]}
    # 需要把 latent 放到 image 键
    def _fb(pol, data, match_keys):
        data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        d2 = {k: v for k, v in data.items() if k != "__LATENT__"}
        d2["image"] = data["__LATENT__"]
        loss = pol.compute_loss({"domain": [DOMAIN], "data": d2})
        params = [q for n, q in pol.named_parameters() if n in set(match_keys)]
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True,
                                    allow_unused=True)
        names = [n for n, q in pol.named_parameters() if n in set(match_keys)]
        return loss, {n: gg for n, gg in zip(names, grads)}

    globals()["_forward_grads"] = _fb
    hist, dt = dlg(pol, g, all_keys, x_hat, x_true, steps=100, lr=0.05,
                   device=device, extra_fixed=fixed)
    cm, cx, rel = corr_metrics(x_hat.detach(), x_true)
    p(f"  ⇒ vision 逐patch相关 mean={cm:.4f} max={cx:.4f}  相对L2={rel:.4f}  "
      f"match {hist[0]:.4f}→{hist[-1]:.4f}  用时 {dt:.1f}s ({dt/100:.2f}s/step)")

    # ---- C: 反演动作序列 ----
    p("\n  [C] 目标 = 动作序列 (1,8,4)")
    y_true = batch["action"].detach()
    y_hat = torch.zeros_like(y_true)
    y_hat.requires_grad_(True)
    fixed2 = {"image": batch["image"], "state": batch["state"]}

    def _fc(pol, data, match_keys):
        data = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        d2 = {k: v for k, v in data.items() if k != "__LATENT__"}
        d2["action"] = data["__LATENT__"]
        loss = pol.compute_loss({"domain": [DOMAIN], "data": d2})
        params = [q for n, q in pol.named_parameters() if n in set(match_keys)]
        grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True,
                                    allow_unused=True)
        names = [n for n, q in pol.named_parameters() if n in set(match_keys)]
        return loss, {n: gg for n, gg in zip(names, grads)}

    globals()["_forward_grads"] = _fc
    hist2, dt2 = dlg(pol, g, all_keys, y_hat, y_true, steps=100, lr=0.05,
                     device=device, extra_fixed=fixed2)
    cos, rel2 = cos_metrics(y_hat.detach(), y_true)
    p(f"  ⇒ action 余弦={cos:.4f}  相对L2={rel2:.4f}  "
      f"match {hist2[0]:.4f}→{hist2[-1]:.4f}  用时 {dt2:.1f}s")

    hr("Part D: 随机基线（什么都没反演出来的水平）")
    rb = torch.randn_like(x_true) * 0.1
    cm0, cx0, rel0 = corr_metrics(rb, x_true)
    p(f"  vision 随机基线: 相关 mean={cm0:.4f}  相对L2={rel0:.4f}")
    ra = torch.zeros_like(y_true)
    cos0, rel20 = cos_metrics(ra, y_true)
    p(f"  action 随机基线: 余弦={cos0:.4f}  相对L2={rel20:.4f}")

    hr("结论")
    p(f"  vision DLG 相关 {cm:.4f} vs 基线 {cm0:.4f}  → 提升 {cm-cm0:+.4f}")
    p(f"  action DLG 余弦 {cos:.4f} vs 基线 {cos0:.4f}  → 提升 {cos-cos0:+.4f}")
    p(f"  单步耗时 ≈ {(dt+dt2)/200:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        p(traceback.format_exc())
    finally:
        try:
            with open("/root/HPT/output/probe3.log", "w", encoding="utf-8") as f:
                f.write(OUT.getvalue())
        except Exception:
            pass
