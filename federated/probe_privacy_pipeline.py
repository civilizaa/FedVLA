#!/usr/bin/env python3
"""探针：核实第4阶段隐私侧的数据管线与模型加载路径（一次性诊断脚本）。

要回答的问题：
  Q1. LocalTrajDataset 的磁盘缓存分支怎么走？为什么之前会撞 env_rollout_fn=None？
  Q2. 真实 batch 的精确形状（image / state / action / language）？
  Q3. init_policy + load_model(model.pth) 能否在无 env 渲染的情况下跑通？
  Q4. 参数量是否 24.97M（MoE 完整）？
"""
import os
import sys
import traceback

sys.path.insert(0, "/root/HPT")

import numpy as np
import torch

ZARR = "/root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100"
GLOBAL_DIR = "/root/HPT/output/p3_moe_eda_cluster/global_r49"


def hr(t):
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


hr("Q1: LocalTrajDataset 磁盘缓存分支")
try:
    import inspect
    from hpt.dataset import local_traj_dataset as ltd
    src = inspect.getsource(ltd.LocalTrajDataset.__init__)
    for i, line in enumerate(src.splitlines()):
        if any(k in line for k in ("use_disk", "regenerate", "env_rollout_fn",
                                   "dataset_path", "zarr", "exists", "load")):
            print(f"  [{i:3d}] {line}")
except Exception:
    traceback.print_exc()


hr("Q2: zarr 真实数据的形状与切片")
import zarr
z = zarr.open(ZARR, mode="r")
d = z["data"]
print("  group keys:", list(z.group_keys()))
for k in d.array_keys():
    a = d[k]
    print(f"  {k:10s} shape={a.shape} dtype={a.dtype}")
print("  meta keys:", list(z["meta"].group_keys()) if "meta" in z.group_keys() else "N/A")


hr("Q2b: 按 LocalTrajDataset.__getitem__ 规则构造 batch")
# observation_horizon=1, action_horizon=8, proprioception_expand=false
idx = 0
img = np.asarray(d["image"][idx])          # (1, 49, 512)  [V=1, P=49, D=512]
print("  image[idx] raw shape:", img.shape)
# __getitem__ 对 image 走 val[:observation_horizon] -> 取 seq 的第 0 个
# sampler 给出 (seq_len, ...)；这里直接取第 0 个时间步
st = np.asarray(d["state"][idx])            # (39,)
ac = np.asarray(d["action"][idx:idx + 8])   # (8, 4)
print("  state[idx] shape:", st.shape)
print("  action[idx:idx+8] shape:", ac.shape)
print("  image 取值范围:", float(img.min()), float(img.max()))
print("  state 取值范围:", float(st.min()), float(st.max()))
print("  action 取值范围:", float(ac.min()), float(ac.max()))

hr("Q3: init_policy + load_model(model.pth) 干跑（无 env）")
try:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from hpt.models.policy import Policy
    from hpt.utils.normalizer import LinearNormalizer

    cfg = OmegaConf.load(os.path.join(GLOBAL_DIR, "config.yaml"))
    cfg = OmegaConf.structured(cfg)

    # ---- 构造 policy（与 init_policy 完全同序）----
    net = OmegaConf.create(dict(cfg.network))
    net["_target_"] = "hpt.models.policy.Policy"
    policy = instantiate(net)
    print("  [1] instantiate(network) OK")

    # update_network_dim：需要一个 dataset 替身
    class DS:
        state_dim = 39
        num_views = 1
        one_action_dim = 4
        action_dim = 4 * 8      # action_multiple_horizon=True -> *8

    from hpt.utils import utils as hutils
    hutils.update_network_dim(cfg, DS(), policy)
    print("  [2] update_network_dim OK")

    policy.init_domain_stem("mujoco_metaworld_task3", cfg.stem)
    print("  [3] init_domain_stem OK")

    # normalizer：从 zarr 拟合（等价于训练时 LinearNormalizer.fit(mode='limits')）
    nz = LinearNormalizer()
    nz.fit(data={"action": np.asarray(d["action"][:]),
                  "state": np.asarray(d["state"][:])},
           last_n_dims=1, mode="limits")
    policy.init_domain_head("mujoco_metaworld_task3", nz, cfg.head)
    print("  [4] init_domain_head OK")

    policy.finalize_modules()
    print("  [5] finalize_modules OK")

    sd = torch.load(os.path.join(GLOBAL_DIR, "model.pth"),
                    map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"  [6] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("      missing[:8]:", missing[:8])
    if unexpected:
        print("      unexpected[:8]:", unexpected[:8])

    n = sum(p.numel() for p in policy.parameters())
    nt = sum(p.numel() for p in policy.trunk.parameters())
    print(f"  [7] 参数量: total={n/1e6:.3f}M  trunk={nt/1e6:.3f}M")
    print(f"      MoE 判定: {'完整' if n > 18e6 else '残缺!!'}")

    hr("Q3b: 前向一次，拿到模型真正期望的 batch 形状")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(dev)

    # 试 (B, T, V, P, D) = (1,1,1,49,512)
    cand = {
        "image": torch.from_numpy(img).float().unsqueeze(0).to(dev),   # (1,1,49,512)
        "state": torch.from_numpy(st).float().unsqueeze(0).to(dev),    # (1,39)
        "action": torch.from_numpy(ac).float().unsqueeze(0).to(dev),   # (1,8,4)
    }
    print("  候选 image shape:", tuple(cand["image"].shape))
    for tag, mut in (("A: (1,1,49,512)", cand),
                     ("B: (1,1,1,49,512)",
                      {**cand, "image": torch.from_numpy(img).float()
                       .unsqueeze(0).unsqueeze(0).to(dev)}),
                     ("C: state (1,1,39)",
                      {**cand, "state": torch.from_numpy(st).float()
                       .unsqueeze(0).unsqueeze(0).to(dev)})):
        try:
            loss = policy.compute_loss({"domain": ["mujoco_metaworld_task3"],
                                         "data": dict(mut)})
            print(f"  [OK] {tag} -> loss={float(loss):.6f}")
        except Exception as e:
            print(f"  [FAIL] {tag}: {type(e).__name__}: {str(e)[:160]}")

    hr("Q4: trunk 梯度（服务器收到的上行更新）")
    policy.train(False)
    for nm, p in policy.named_parameters():
        p.requires_grad_(nm.startswith("trunk."))
    loss = policy.compute_loss({"domain": ["mujoco_metaworld_task3"],
                                 "data": dict(cand)})
    loss.backward()
    g = {n: p.grad.detach().clone() for n, p in policy.named_parameters()
         if n.startswith("trunk.") and p.grad is not None}
    tot = sum(v.numel() for v in g.values())
    print(f"  trunk 梯度张量数={len(g)} 参数量={tot} ({tot/1e6:.3f}M)")
    print(f"  前 3 个键: {list(g)[:3]}")
    if g:
        k0 = list(g)[0]
        print(f"  {k0} 范数={float(g[k0].norm()):.6e} 形状={tuple(g[k0].shape)}")

except Exception:
    traceback.print_exc()
