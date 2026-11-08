#!/usr/bin/env python3
"""FedVLA 第 4 阶段隐私侧：DP-SGD 上行加噪 + 梯度反演攻击（路线图 5.2）。

## 回答三个问题

1. **上行更新泄漏了多少隐私？** → 逐通路敏感度（消融法 + 梯度反演）
2. **DP 噪声能挡住多少？** → ε 扫描下的重建质量
3. **H3：统一 DP 预算劣于模态感知预算？** → 相同总 ε 下两套分配的直接对比

## 威胁模型（论文 6.3）

- 服务器**诚实但好奇**：遵守协议，但会分析收到的上行更新。
- 攻击者能力：拿到**被 DP 噪声污染后的**上行更新（FedVLA 的上行 =
  `trunk.*` + `normalizer.<domain>` + `modalities_tokens.*`），
  用它反演客户端的私有观测（视觉特征 / 状态）与动作。
- 攻击方法：**DLG（Deep Leakage from Gradients）**，联合/分别优化候选潜变量，
  使候选上的 trunk 梯度尽量接近服务器收到的梯度。

## ⚠️ 三个与架构强相关的事实（2026-11-08 实测，不是推测）

### A. 反演目标是 ResNet 特征图，不是像素
Meta-World 的 zarr 里 `image` 是 **512 维 ResNet 特征**（形状 `(N,1,49,512)`，
取值 [0, 15.77]），不是 512×512 像素。所以：
- 潜变量 x̂ 形状 `(1,1,49,512)`，**不用 sigmoid**；
- 指标是**逐 patch Pearson 相关** + 相对 L2；
- 出图是 7×7 空间 patch 热力图（对 512 通道取均值），不是照片。

### B. 梯度匹配必须取**交集**
MoE 的 top-k 路由**随输入变化** ⇒ 候选前向的梯度张量集合是服务器收到的
梯度集合的真子集（实测 15.82M vs 17.13M）。攻击者只能用交集匹配。
默认 `match_mode="dense"`（只匹配 blocks 0-7 的稠密参数），既稳定又快。

### C. 语言通路在本配置下**不存在**
实测 `policy.modalities = ['image','state']`，且 `loss(有语言) − loss(无语言) = 0`
（差 0.000e+00）⇒ HPT 的 T5 语言嵌入没有被策略消费。
故模态敏感度分析覆盖 **视觉 / 状态 / 动作** 三通路，论文如实写。

## 模态感知预算为什么在 FedVLA 里特别有效（论文的机制解释）

FedVLA 的上行里 **stems/heads 永不上行** ⇒ 上行中的"模态专属"部分只剩
`modalities_tokens.{image,state}`（各 256 参数）与 `normalizer` 的统计量。
并行组合语义下总 ε = max_m ε_m，于是：
- 模态感知分配 ε_m = ε_total · w_m/w_max ⇒ vision 拿满 ε_total，
  **shared（=真正泄漏的 trunk）只拿 0.5 ε_total ⇒ σ 更大 ⇒ 保护更强**；
- 统一预算则给 trunk 全额 ε_total ⇒ 噪声更小 ⇒ 泄漏更多。
⇒ **在相同总预算下，模态感知对 FedVLA 实际泄漏通道的保护强于统一预算**（H3）。

## 用法

```bash
# 干跑（1 个样本、20 步）
python federated/privacy_experiment.py --samples 1 --inv-steps 20 --epsilons 8 --modes uniform

# 完整扫描
bash federated/run_privacy_scan.sh
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, "/root/HPT")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# ⚠️ DLG 的硬前提：必须用 **数学后端** 的注意力。
#    PyTorch 默认走 `aten::_scaled_dot_product_efficient_attention`，
#    它的 backward **没有二阶导实现** ⇒ DLG 的 create_graph=True 直接报
#    "derivative for aten::_scaled_dot_product_efficient_attention_backward
#     is not implemented"。必须全局关掉 flash / mem-efficient，强制 math。
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

from dp import (DEFAULT_MODALITY_WEIGHTS, DPConfig, allocate_by_sigma,
                allocate_modality_budget, apply_dp_to_update, effective_snr,
                epsilon_from_sigma, group_by_modality, is_shared_uplink_key)
from inversion import invert_observation, rel_l2

DEFAULT_MODEL_DIR = "/root/HPT/output/p3_moe_eda_cluster/global_r49"
DEFAULT_ZARR = "/root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100"
DEFAULT_DOMAIN = "mujoco_metaworld_task3"
DEFAULT_OUT = "/root/HPT/output/privacy"

#: 完整 MoE policy 的参数量下限（实测：完整 22.380M / 残缺 13.94M）
MOE_PARAM_FLOOR = 18e6

#: 各预算方案的**模态敏感度权重**（ε_m ∝ w_m，并行组合下 max_m ε_m = 总 ε）
#:
#: · `uniform`     —— 不分模态，所有通路同一个 σ
#: · `modality`    —— **先验**权重（dp.py 的 DEFAULT_MODALITY_WEIGHTS）：
#:   视觉最敏感 → 拿满额预算、动作噪声最大。
#:   ⚠️ 这套权重是**事前**定的（路线图 5.2 原始设定），不因实验结果调整。
#: · `modality_rev` —— **按实测可攻击性**重新分配（事后分析）：
#:   消融法测得的"方向泄漏强度 1−cos"排序是 action 1.358 > image 0.760 > state 0.136，
#:   与先验假设（视觉最敏感）**相反**。这个方案回答：
#:   "若按实测可攻击性分配预算，保护能强多少？"
#:   shared 权重压到 0.10 ⇒ 外传的 trunk 只拿 0.2ε_total ⇒ σ 显著更大。
EXTRA_MODE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "modality_rev": {"vision": 0.30, "state": 0.10, "action": 0.50, "shared": 0.10},
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# 1. 建模：绕开 LocalTrajDataset，直接复刻 hpt.run.init_policy 的顺序
# ---------------------------------------------------------------------------
def build_policy(model_dir: str, zarr_path: str, domain: str, device):
    """构造 policy 并载入 global_rNN 的权重。

    为什么不走 `hpt.run.init_policy(cfg, dataset, ...)`：
    它强制要求一个真实的 dataset 对象（state_dim / num_views / one_action_dim /
    action_dim / get_normalizer()），而 `LocalTrajDataset` 在没有 env 渲染函数时
    会退化成「每次重渲染 16 分钟」。这里用 **shim dataset + zarr 直读** 复刻同样的
    初始化顺序，语义与训练时完全一致。
    """
    import zarr
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from hpt.models.policy import Policy
    from hpt.utils import utils as hutils
    from hpt.utils.normalizer import LinearNormalizer

    cfg = OmegaConf.structured(OmegaConf.load(os.path.join(model_dir, "config.yaml")))
    net = OmegaConf.create(dict(cfg.network))
    net["_target_"] = "hpt.models.policy.Policy"
    policy = instantiate(net)

    z = zarr.open(zarr_path, mode="r")
    d = z["data"]

    class _DS:
        """shim dataset：只提供 init_policy 需要的四个维度。"""
        state_dim = 39            # zarr state (N, 39)
        num_views = 1             # zarr image (N, 1, 49, 512)
        one_action_dim = 4        # zarr action (N, 4)
        action_dim = 4 * 8        # action_multiple_horizon=True → ×8 = 32

    hutils.update_network_dim(cfg, _DS(), policy)
    policy.init_domain_stem(domain, cfg.stem)

    nz = LinearNormalizer()
    nz.fit(data={"action": np.asarray(d["action"][:]),
                  "state": np.asarray(d["state"][:])},
           last_n_dims=1, mode="limits")
    policy.init_domain_head(domain, nz, cfg.head)
    policy.finalize_modules()

    sd = torch.load(os.path.join(model_dir, "model.pth"),
                    map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    policy.eval()                       # 关 dropout ⇒ 前向确定性
    policy.to(device)

    n_all = sum(q.numel() for q in policy.parameters())
    n_tr = sum(q.numel() for q in policy.trunk.parameters())
    log(f"[build] missing={len(missing)} unexpected={len(unexpected)}（unexpected 是其它域的 stems/heads，属预期）")
    log(f"[build] total={n_all/1e6:.3f}M  trunk={n_tr/1e6:.3f}M")
    if n_all < MOE_PARAM_FLOOR:
        raise RuntimeError(
            f"参数量 {n_all/1e6:.3f}M < {MOE_PARAM_FLOOR/1e6:.0f}M ⇒ MoE 结构缺失！"
            "请检查 FEDVLA_MOE / FEDVLA_MOE_EXPERTS / FEDVLA_MOE_TOPK / "
            "FEDVLA_MOE_EXPERT_RATIO / FEDVLA_MOE_LAYERS 是否逐字复刻训练侧设置")
    log(f"[build] MoE 完整性校验通过（>{MOE_PARAM_FLOOR/1e6:.0f}M）")
    log(f"[build] policy.modalities = {getattr(policy, 'modalities', None)}")
    return policy, cfg, d


# ---------------------------------------------------------------------------
# 2. 数据
# ---------------------------------------------------------------------------
def make_batch(d, idx: int, device, action_horizon: int = 8):
    """按 LocalTrajDataset.__getitem__ 的切片规则构造真实 batch。

    实测规则（hpt/dataset/local_traj_dataset.py）：
      非 action 键：`val[:observation_horizon]`，observation_horizon=1
      action：      `val[observation_horizon-1 : action_horizon+observation_horizon-1]` = [0:8]
      proprioception_expand=false ⇒ state 原样 (39,)
    """
    img = torch.from_numpy(np.asarray(d["image"][idx])).float()[None].to(device)      # (1,1,49,512)
    st = torch.from_numpy(np.asarray(d["state"][idx])).float()[None, None].to(device)  # (1,1,39)
    ac = torch.from_numpy(np.asarray(d["action"][idx:idx + action_horizon])).float()[None].to(device)
    return {"image": img, "state": st, "action": ac}


# ---------------------------------------------------------------------------
# 3. 上行更新与通路敏感度
# ---------------------------------------------------------------------------
def compute_local_grad(policy, domain: str, batch: Dict) -> Dict:
    """算客户端**本地**的梯度（全部参数）。

    ⚠️ DP 作用在**本地全部参数**上，而不是只作用在上行里。理由（实测）：
       FedVLA 的上行 = `trunk.*` + `normalizer.<domain>` + `modalities_tokens.*`，
       但实测发现：
         · `modalities_tokens` **不参与前向**（没有梯度）⇒ 上行里没有视觉组参数；
         · `normalizer` 只有 action / state 两组统计量在图里；
       ⇒ 如果只对上行加噪，"模态感知预算"在 FedVLA 上是**空的**（只有一个组）。
       DP-SGD 的本意是保护**客户端本地训练过程**里的隐私（stems/heads 也可能是
       私有信息，且恶意服务器可通过多轮交互推断它们），
       服务器最终看到的上行是这份本地更新的 **trunk 子集**。
       这样模态分组才有意义，也才和 dp.py / test_privacy.py 的语义一致。
    """
    for n_, q in policy.named_parameters():
        q.requires_grad_(True)
    data = {k: v.clone() for k, v in batch.items()}
    loss = policy.compute_loss({"domain": [domain], "data": data})
    loss.backward()
    grads = {n_: q.grad.detach().clone() for n_, q in policy.named_parameters()
             if q.grad is not None}
    policy.zero_grad(set_to_none=True)
    return grads


def split_uplink(local_grad: Dict) -> Tuple[Dict, Dict]:
    """把本地梯度拆成 (上行共享部分, trunk 部分)。"""
    uplink = {k: v for k, v in local_grad.items() if is_shared_uplink_key(k)}
    trunk = {k: v for k, v in uplink.items() if k.startswith("trunk.")}
    return uplink, trunk


def modality_sensitivity(policy, domain: str, batch: Dict, donor: Dict,
                          base: Optional[Dict] = None) -> Dict[str, float]:
    """**消融法**测逐通路敏感度（论文的量化口径）。

    对通路 m，把该通路的输入换成**另一个样本**的（donor），
    其余通路保持真实，重新算梯度 g_m。
    依赖度 D_m = ‖g_full − g_m‖₂ / ‖g_full‖₂
    ⇒ D_m 越大，说明上行更新越"依赖"该通路的私有内容，即该通路泄漏越多。

    ⚠️ base 必须是**原始 batch**的梯度（g_full）。若误用"已替换过 image 的
    梯度"当 base，image 那一路的 D 恒为 0 —— 已在 2026-11-08 修正。

    返回：{"image": D, "state": D, "action": D, "_per_group": {...}}
    """
    if base is None:
        base = compute_local_grad(policy, domain, batch)
    out: Dict[str, float] = {}
    per_group: Dict[str, Dict[str, float]] = {}
    cos_overall: Dict[str, float] = {}
    for mod in ("image", "state", "action"):
        alt = dict(batch)
        alt[mod] = donor[mod].clone()
        g = compute_local_grad(policy, domain, alt)
        num = 0.0
        den = 0.0
        dot = 0.0
        gm: Dict[str, float] = {}
        for grp, keys in group_by_modality(g).items():
            n2 = 0.0
            d2 = 0.0
            dt = 0.0
            for k in keys:
                if k not in base:
                    continue
                n2 += float(((base[k] - g[k]) ** 2).sum())
                d2 += float((base[k] ** 2).sum())
                dt += float((base[k] * g[k]).sum())
            gm[grp] = float(np.sqrt(n2) / (np.sqrt(d2) + 1e-12)) if d2 > 0 else 0.0
            num += n2
            den += d2
            dot += dt
        out[mod] = float(np.sqrt(num) / (np.sqrt(den) + 1e-12))
        # 余弦相似度：把该通路换成别的样本后，梯度**方向**还剩多少重合。
        # 泄漏强度 = 1 − cos ∈ [0, 2]（0 = 完全不依赖该通路，2 = 完全依赖）。
        cos_overall[mod] = float(dot / (np.sqrt(den) * np.sqrt(num + den - 2 * dot) + 1e-12))
        per_group[mod] = gm
    out["_per_group"] = per_group      # type: ignore[assignment]
    out["_cos"] = cos_overall            # type: ignore[assignment]
    return out


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def run(args) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, "privacy_results.json")

    log(f"[run] device={device}")
    policy, cfg, d = build_policy(args.model_dir, args.zarr, args.domain, device)

    epsilons = [float(x) for x in args.epsilons.split(",") if x]
    sigmas = [float(x) for x in args.sigmas.split(",") if x]
    modes = [m for m in args.modes.split(",") if m]
    targets = [t for t in args.targets.split(",") if t]
    samples = [int(x) for x in args.samples.split(",") if x]

    all_results: List[Dict] = []
    if os.path.exists(results_path) and not args.force:
        try:
            prev = json.load(open(results_path, encoding="utf-8"))
            all_results = prev.get("runs", [])
            log(f"[run] 已有 {len(all_results)} 条结果，续跑（--force 可清空）")
        except Exception:
            all_results = []

    done_keys = {(r["sample"], r["mode"], r.get("sigma_ref"), r["target"])
                 for r in all_results}

    for si, idx in enumerate(samples):
        batch = make_batch(d, idx, device)
        # ⚠️ donor 必须取**跨 episode** 的样本。
        #    zarr 里 7204 步 = 100 集 × ~72 步，**相邻索引是同一条轨迹的相邻时间步**，
        #    动作几乎完全相同 ⇒ 消融法测出来的 D_action 会是 0（2026-11-08 实测到 0.0）。
        #    取一半数据集之外 ⇒ 一定是另一条轨迹。
        n_total = int(d["action"].shape[0])
        donor_idx = (idx + n_total // 2) % n_total
        donor = make_batch(d, donor_idx, device)
        log(f"[sample] idx={idx}  donor_idx={donor_idx}  "
            f"Δaction_rel={float(torch.linalg.vector_norm(batch['action']-donor['action'])/(torch.linalg.vector_norm(batch['action'])+1e-12)):.4f}  "
            f"Δimage_rel={float(torch.linalg.vector_norm(batch['image']-donor['image'])/(torch.linalg.vector_norm(batch['image'])+1e-12)):.4f}  "
            f"Δstate_rel={float(torch.linalg.vector_norm(batch['state']-donor['state'])/(torch.linalg.vector_norm(batch['state'])+1e-12)):.4f}")

        log(f"\n===== 样本 {si+1}/{len(samples)} (zarr idx={idx}) =====")

        # ---- 4.1 干净本地梯度 + 上行拆分 ----
        local_grad = compute_local_grad(policy, args.domain, batch)
        uplink, trunk_grad = split_uplink(local_grad)
        groups = group_by_modality(local_grad)
        log(f"[local] 本地梯度：{len(local_grad)} 张量 / "
            f"{sum(v.numel() for v in local_grad.values())/1e6:.3f}M 参数")
        for m, ks in sorted(groups.items()):
            log(f"          组 {m:8s}: {len(ks):4d} 张量, "
                f"{sum(local_grad[k].numel() for k in ks)/1e6:8.4f}M 参数")
        log(f"[uplink] 其中上行的共享部分：{len(uplink)} 张量 / "
            f"{sum(v.numel() for v in uplink.values())/1e6:.3f}M 参数")
        log(f"[uplink] 攻击者可见的 trunk 梯度：{len(trunk_grad)} 张量 / "
            f"{sum(v.numel() for v in trunk_grad.values())/1e6:.3f}M 参数")

        # ---- 4.2 逐通路敏感度（消融）----
        t0 = time.time()
        sens = modality_sensitivity(policy, args.domain, batch, donor, base=local_grad)
        per_g = sens.pop("_per_group", {})
        cos_g = sens.pop("_cos", {})
        log(f"[sens] 通路敏感度 D_m = {json.dumps({k: round(v, 4) for k, v in sens.items()})}"
            f"  ({time.time()-t0:.1f}s)")
        log(f"[sens] 梯度方向余弦（换掉该通路后）= "
            f"{json.dumps({k: round(v, 4) for k, v in cos_g.items()})}"
            f"  泄漏强度=1−cos: "
            f"{json.dumps({k: round(1-v, 4) for k, v in cos_g.items()})}")
        for mod, gm in per_g.items():
            log(f"          {mod:7s} 分组变化: " +
                json.dumps({k: round(v, 4) for k, v in gm.items()}))
        all_results.append({
            "kind": "sensitivity", "sample": idx, "mode": "none", "eps": None,
            "sigma_ref": None, "target": "all", "sensitivity": sens,
            "sensitivity_per_group": per_g, "sensitivity_cos": cos_g,
        })

        # ---- 4.3 无噪声参考反演（攻击者的上界）----
        for tgt in targets:
            key = (idx, "clean", None, tgt)
            if key in done_keys:
                continue
            t0 = time.time()
            res = invert_observation(
                policy, args.domain, batch, trunk_grad, target=tgt,
                steps=args.inv_steps, lr=args.inv_lr, device=device,
                match_mode=args.match_mode, seed=args.seed, verbose=args.verbose)
            rec = _pack(res, idx, "clean", None, tgt, save_recon=(idx == samples[0]))
            all_results.append(rec)
            log(f"[inv] clean/{tgt}: {_metric_str(res)} ({time.time()-t0:.0f}s)")

        # ---- 4.4 σ 扫描 × 预算方案 ----
        # ⚠️ 自变量是**噪声倍率 σ**而不是 ε，原因见 dp.epsilon_from_sigma 的文档：
        #    trunk 有 17.1M 参数 ⇒ 噪声总范数 = σ·C·√17.1M。σ=1（ε≈1）时噪声范数
        #    4139 ≫ 信号范数 1 ⇒ 所有 ε∈[1,32] 都在噪声地板上，曲线全平。
        #    真正的转折在 σ≈1e-4（对应 ε≈1e5~1e7）。
        #    两种模式用**同一个 σ_ref** ⇒ 各自的总 ε 相同 ⇒ H3 对比是等预算的。
        for sig in sigmas:
            for mode in modes:
                alloc = allocate_by_sigma(DPConfig(
                    delta=args.delta, steps=args.dp_steps,
                    num_clients=args.clients,
                    modality_aware=mode.startswith("modality"),
                    modality_weights=EXTRA_MODE_WEIGHTS.get(
                        mode, DEFAULT_MODALITY_WEIGHTS)), sig)
                # DP 作用在**本地全部参数**上；服务器看到的是它的 trunk 子集。
                noisy_local = apply_dp_to_update(local_grad, alloc,
                                                 clip_norm=args.clip_norm,
                                                 seed=args.seed)
                noisy_trunk = {k: v for k, v in noisy_local.items()
                               if k.startswith("trunk.")}
                snr = effective_snr(local_grad, alloc, clip_norm=args.clip_norm)
                # **学习信号保真度**：cos(干净梯度, 加噪梯度)。
                # FedAvg 的收敛方向完全取决于上行更新的方向，
                # 所以这个量直接对应"噪声吃掉了多少学习信号"（效用的低成本代理）。
                fid: Dict[str, float] = {}
                for grp, keys in group_by_modality(local_grad).items():
                    dot = 0.0
                    na = 0.0
                    nb = 0.0
                    for k in keys:
                        a = local_grad[k].reshape(-1).float()
                        b = noisy_local[k].reshape(-1).float()
                        dot += float((a * b).sum())
                        na += float((a ** 2).sum())
                        nb += float((b ** 2).sum())
                    fid[grp] = float(dot / (np.sqrt(na) * np.sqrt(nb) + 1e-12))
                log(f"\n[dp] σ_ref={sig:g} mode={mode}: 总ε={alloc.achieved_epsilon:.4g} σ=" +
                    json.dumps({k: round(v, 6) for k, v in alloc.sigma_by_modality.items()}) +
                    "  SNR=" + json.dumps({k: round(v, 4) for k, v in snr.items()}))
                for tgt in targets:
                    k2 = (idx, mode, sig, tgt)
                    if k2 in done_keys:
                        continue
                    t0 = time.time()
                    res = invert_observation(
                        policy, args.domain, batch, noisy_trunk, target=tgt,
                        steps=args.inv_steps, lr=args.inv_lr, device=device,
                        match_mode=args.match_mode, seed=args.seed,
                        verbose=args.verbose)
                    rec = _pack(res, idx, mode, alloc.achieved_epsilon, tgt,
                                 save_recon=(idx == samples[0]))
                    rec["sigma_ref"] = sig
                    rec["sigma_by_modality"] = alloc.sigma_by_modality
                    rec["eps_by_modality"] = alloc.eps_by_modality
                    rec["achieved_epsilon"] = alloc.achieved_epsilon
                    rec["snr_by_modality"] = snr
                    rec["fidelity_by_group"] = fid
                    all_results.append(rec)
                    # ⚠️ 每条结果**立刻**落盘：AutoDL 实例会被关机，
                    #    只在样本末尾保存会整段丢失（实测过好几次）。
                    _save(results_path, all_results, args)
                    log(f"[inv] σ={sig:g}/{mode}/{tgt}: {_metric_str(res)} "
                        f"(ε={alloc.achieved_epsilon:.4g}) "
                        f"保真度={json.dumps({k: round(v, 3) for k, v in fid.items()})} "
                        f"({time.time()-t0:.0f}s)")

        # ---- 4.5 增量落盘（防实例关机丢结果）----
        _save(results_path, all_results, args)

    _save(results_path, all_results, args)
    log(f"\n[done] 共 {len(all_results)} 条结果 → {results_path}")
    return {"runs": all_results}


def _pack(res: Dict, sample: int, mode: str, eps: Optional[float], target: str,
          save_recon: bool = True) -> Dict:
    rec = {
        "kind": "inversion", "sample": sample, "mode": mode, "eps": eps,
        "target": target,
        "match_loss_final": res.get("final_match_loss"),
        "match_loss_best": res.get("best_match_loss"),
        "rel_l2": res.get("rel_l2"),
        "baseline_rel_l2": res.get("baseline_rel_l2"),
        "n_common_keys": res.get("n_common_keys"),
        "seconds": res.get("seconds"),
    }
    for k in ("corr_mean", "corr_max", "baseline_corr_mean", "baseline_corr_max",
              "cos", "baseline_cos", "psnr", "ssim", "baseline_psnr",
              "baseline_ssim", "psnr_feature"):
        if k in res:
            rec[k] = res[k]
    # 重建张量（用于出图）。体积大：image 目标每次 25088 个浮点 ⇒
    # 只保留**第一个样本**的重建，其余样本只留指标。
    if not save_recon:
        return rec
    if target == "image":
        rec["recon_image"] = np.asarray(res["recon"], dtype=np.float32).tolist()
        rec["real_image"] = np.asarray(res["real"], dtype=np.float32).tolist()
    elif target == "action":
        rec["recon_action"] = np.asarray(res["recon"], dtype=np.float32).tolist()
        rec["real_action"] = np.asarray(res["real"], dtype=np.float32).tolist()
    else:
        rec["recon_state"] = np.asarray(res["recon"], dtype=np.float32).tolist()
        rec["real_state"] = np.asarray(res["real"], dtype=np.float32).tolist()
    return rec


def _metric_str(res: Dict) -> str:
    if "corr_mean" in res:
        return (f"corr={res['corr_mean']:.4f} (基线 {res['baseline_corr_mean']:.4f}) "
                f"rel_l2={res['rel_l2']:.4f}")
    if "cos" in res:
        return (f"cos={res['cos']:.4f} (基线 {res['baseline_cos']:.4f}) "
                f"rel_l2={res['rel_l2']:.4f}")
    return f"rel_l2={res.get('rel_l2', float('nan')):.4f}"


def _save(path: str, runs: List[Dict], args) -> None:
    payload = {
        "runs": runs,
        "meta": {
            "model_dir": args.model_dir, "zarr": args.zarr, "domain": args.domain,
            "sigmas": args.sigmas, "epsilons": args.epsilons,
            "modes": args.modes, "targets": args.targets,
            "samples": args.samples, "delta": args.delta, "dp_steps": args.dp_steps,
            "clients": args.clients, "clip_norm": args.clip_norm,
            "inv_steps": args.inv_steps, "inv_lr": args.inv_lr,
            "match_mode": args.match_mode,
        },
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="FedVLA 隐私侧实验（DP 加噪 + 梯度反演）")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--zarr", default=DEFAULT_ZARR)
    ap.add_argument("--domain", default=DEFAULT_DOMAIN)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epsilons", default="", help="（保留接口）按 ε 扫描；优先用 --sigmas")
    ap.add_argument("--sigmas", default="1e-4,1e-3,1e-2,1e-1,1",
                    help="噪声倍率 σ 扫描（对数间隔）。⚠️ 不用 ε 作自变量："
                         "trunk 17.1M 参数 ⇒ 噪声范数 = σ·C·√17.1M，"
                         "ε∈[1,32] 全在噪声地板上，曲线是平的")
    ap.add_argument("--modes", default="uniform,modality")
    ap.add_argument("--targets", default="image,action,state")
    ap.add_argument("--samples", default="0,1,2")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--dp-steps", type=int, default=25)
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--inv-steps", type=int, default=200)
    ap.add_argument("--inv-lr", type=float, default=0.05)
    ap.add_argument("--match-mode", default="dense", choices=["dense", "all"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略已有结果重跑")
    args = ap.parse_args(argv)

    try:
        run(args)
    except Exception:
        tb = traceback.format_exc()
        log(tb)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
