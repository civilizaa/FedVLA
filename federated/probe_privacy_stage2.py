#!/usr/bin/env python3
"""探针 stage2：补 Q4（trunk 梯度）+ 摸清 trunk 键结构（DP 模态分组的依据）。

背景（2026-11-08 探针 stage1 结论）：
  - model.pth 加载后 total=22.380M / trunk=21.082M -> MoE 完整
  - 正确 batch 布局：image (B,1,49,512) / state (B,1,39) / action (B,8,4)，loss 可算出
  - image 是 ResNet 特征（49 patch x 512 维），不是像素

本脚本要回答：
  Q4. trunk 梯度（服务器收到的上行更新）张量数/范数  —— stage1 被截断
  Q5. trunk 的 488 个键怎么组织？哪部分属于"视觉/语言/动作"通路？
  Q6. modalities_tokens / normalizer 的键与形状（上行链路里的非 trunk 部分）
  Q7. language 是否需要参与前向？
  Q8. /root/HPT/data 下是否有原始像素数据集（那样能画真正的照片级重建图）
"""
import io
import os
import sys
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


def main():
    import zarr
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from hpt.models.policy import Policy
    from hpt.utils import utils as hutils
    from hpt.utils.normalizer import LinearNormalizer

    z = zarr.open(ZARR, mode="r")
    d = z["data"]

    cfg = OmegaConf.structured(OmegaConf.load(os.path.join(GLOBAL_DIR, "config.yaml")))
    net = OmegaConf.create(dict(cfg.network))
    net["_target_"] = "hpt.models.policy.Policy"
    policy = instantiate(net)

    class DS:
        state_dim = 39
        num_views = 1
        one_action_dim = 4
        action_dim = 32

    hutils.update_network_dim(cfg, DS(), policy)
    policy.init_domain_stem(DOMAIN, cfg.stem)

    nz = LinearNormalizer()
    nz.fit(data={"action": np.asarray(d["action"][:]), "state": np.asarray(d["state"][:])},
           last_n_dims=1, mode="limits")
    policy.init_domain_head(DOMAIN, nz, cfg.head)
    policy.finalize_modules()

    sd = torch.load(os.path.join(GLOBAL_DIR, "model.pth"), map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    p(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    n_all = sum(q.numel() for q in policy.parameters())
    n_tr = sum(q.numel() for q in policy.trunk.parameters())
    p(f"[load] total={n_all/1e6:.3f}M trunk={n_tr/1e6:.3f}M "
      f"MoE={'完整' if n_all > 18e6 else '残缺!!'}")

    # ---------------- Q5: trunk 键结构 ----------------
    hr("Q5: trunk 参数键结构（按前 3 段前缀聚合）")
    agg = {}
    for nm, q in policy.trunk.named_parameters():
        parts = nm.split(".")
        key = ".".join(parts[:3]) if len(parts) >= 3 else nm
        a = agg.setdefault(key, {"n": 0, "cnt": 0, "shape": None})
        a["n"] += q.numel()
        a["cnt"] += 1
        if a["shape"] is None:
            a["shape"] = tuple(q.shape)
    for k in sorted(agg, key=lambda k: -agg[k]["n"]):
        v = agg[k]
        p(f"  {k:52s} cnt={v['cnt']:3d} numel={v['n']:>9d} e.g.shape={v['shape']}")
    p(f"  -- trunk 前缀分组数={len(agg)} 总参数={sum(v['n'] for v in agg.values())}")

    hr("Q5b: transformer block 内部结构（看模态专属参数）")
    # 打印第 0 个 block 的全部参数名
    blk_names = [nm for nm, _ in policy.trunk.named_parameters() if ".blocks.0." in nm]
    p(f"  blocks.0 参数数={len(blk_names)}")
    for nm in blk_names:
        q = dict(policy.trunk.named_parameters())[nm]
        p(f"    {nm:70s} {tuple(q.shape)}")
    # 顶层（block 之外）参数
    p("  -- 非 blocks.0 的 trunk 顶层参数 --")
    for nm, q in policy.trunk.named_parameters():
        if ".blocks." not in nm:
            p(f"    {nm:70s} {tuple(q.shape)}")

    hr("Q6: modalities_tokens / normalizer 键")
    for nm, q in policy.named_parameters():
        if nm.startswith("modalities_tokens") or nm.startswith("normalizer"):
            p(f"  {nm:60s} {tuple(q.shape)} numel={q.numel()}")

    # ---------------- 构造 batch ----------------
    idx = 0
    img = torch.from_numpy(np.asarray(d["image"][idx])).float().unsqueeze(0)      # (1,1,49,512)
    st = torch.from_numpy(np.asarray(d["state"][idx])).float().unsqueeze(0).unsqueeze(0)  # (1,1,39)
    ac = torch.from_numpy(np.asarray(d["action"][idx:idx + 8])).float().unsqueeze(0)       # (1,8,4)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(dev)
    img, st, ac = img.to(dev), st.to(dev), ac.to(dev)

    hr("Q7: language 是否参与前向")
    for tag, extra in (("no-language", {}),
                       ("language (1,1,768)",
                        {"language": torch.from_numpy(
                            np.asarray(d["language"][idx])[:1]).float().unsqueeze(0).to(dev)})):
        try:
            data = {"image": img, "state": st, "action": ac, **extra}
            loss = policy.compute_loss({"domain": [DOMAIN], "data": data})
            p(f"  [OK] {tag} -> loss={float(loss):.6f}")
        except Exception as e:
            p(f"  [FAIL] {tag}: {type(e).__name__}: {str(e)[:200]}")

    # ---------------- Q4: trunk 梯度 ----------------
    hr("Q4: trunk 梯度（= 服务器收到的上行更新）")
    policy.train(False)
    for nm, q in policy.named_parameters():
        q.requires_grad_(nm.startswith("trunk."))
    data = {"image": img, "state": st, "action": ac}
    loss = policy.compute_loss({"domain": [DOMAIN], "data": data})
    p(f"  loss={float(loss):.6f}")
    loss.backward()
    g = {nm: q.grad.detach().clone() for nm, q in policy.named_parameters()
         if nm.startswith("trunk.") and q.grad is not None}
    tot = sum(v.numel() for v in g.values())
    p(f"  trunk 梯度张量数={len(g)} 参数量={tot} ({tot/1e6:.3f}M)")
    if g:
        ks = list(g)
        p(f"  前 3 个键: {ks[:3]}")
        for k in ks[:3]:
            p(f"    {k:60s} shape={tuple(g[k].shape)} norm={float(g[k].norm()):.6e}")
        p(f"  全局 L2 范数={float(torch.sqrt(sum((v.double()**2).sum() for v in g.values()))):.6e}")
        p(f"  平均每张量 L2={float(torch.sqrt(sum((v.double()**2).sum() for v in g.values())))/max(len(g),1):.6e}")

    # ---------------- Q8: 是否有原始像素数据集 ----------------
    hr("Q8: /root/HPT/data 目录清单（找原始像素数据集）")
    base = "/root/HPT/data"
    if os.path.isdir(base):
        for e in sorted(os.listdir(base)):
            full = os.path.join(base, e)
            if os.path.isdir(full):
                try:
                    import zarr as _z
                    zt = _z.open(full, mode="r")
                    keys = list(zt.group_keys())
                    sub = []
                    if "data" in keys:
                        for k in zt["data"].array_keys():
                            sub.append(f"{k}{tuple(zt['data'][k].shape)}")
                    p(f"  {e}: keys={keys} data={sub}")
                except Exception:
                    p(f"  {e}: (非 zarr 或不可读) {sorted(os.listdir(full))[:6]}")
            else:
                p(f"  {e} (file)")
    else:
        p("  /root/HPT/data 不存在")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        p(tb)
    finally:
        try:
            with open("/root/HPT/output/probe2.log", "w", encoding="utf-8") as f:
                f.write(OUT.getvalue())
        except Exception:
            pass
