"""[FedVLA-BASE] FedProx / SCAFFOLD 的训练循环钩子。

由 federated/patch_hpt_fedbaseline.py 复制到 hpt/utils/fedvla_baseline.py。

═══════════════════════════════════════════════════════════════════════════════
设计原则
═══════════════════════════════════════════════════════════════════════════════
* **默认完全关闭**：所有开关靠环境变量，关闭时本模块不做任何事 ⇒
  训练行为与不打补丁逐字节一致（铁律 D：跨批次实验必须同构）。
* **不改 HPT 的模型/损失**：只在 backward() 之后改 .grad，
  数学上等价于把修正项并入 loss（∂[(μ/2)‖w−w_g‖²]/∂w = μ(w−w_g)）。
* **不自造依赖**：只用 torch + 标准库；FedProx/SCAFFOLD 的数学在
  federated/fed_baselines.py 里（可离线单测），本文件只做"接线"。

═══════════════════════════════════════════════════════════════════════════════
键空间（2026-11-22 踩坑，必须记住）
═══════════════════════════════════════════════════════════════════════════════
HPT 的 `Policy.trunk` 是 ModuleDict ⇒ **模型参数名**是 `trunk.trunk.xxx`；
而聚合写出的 `trunk.pth` 的键是 `trunk.xxx`（aggregate.extract_trunk_only
剥掉了第一层 "trunk." 前缀）。

  ⚠️ 两者**不一致**！若直接拿 trunk.pth 当锚点/控制变量，
     键对不上 ⇒ 修正项静默作用不到任何参数 ⇒ 基线退化成 FedAvg 且**零报错**。

⇒ 约定：**基线的锚点/控制变量一律存在"全模型键空间"**（`trunk.trunk.xxx`），
   由 run_fedavg.sh 在写盘时把 trunk.pth 的键补回前缀。
   本模块按模型参数名匹配 ⇒ 三方（锚点 / 控制变量 / 模型参数）键空间一致。

═══════════════════════════════════════════════════════════════════════════════
作用范围：只覆盖 trunk.*
═══════════════════════════════════════════════════════════════════════════════
  * 服务器每轮聚合的主体参数就是 trunk（average_trunk）；
  * normalizer / modalities_tokens 虽属上行共享参数，但走"同名同形才平均"，
    跨异构客户端形状不一致，且第 0 轮的预训练侧没有对应文件 ⇒
    纳入会造成**跨轮口径不一致**（违反铁律 D）⇒ 排除。
"""
from __future__ import annotations

import os

import torch

# ---- 文件缓存（关键：不能每个 step 都 torch.load 50MB 锚点）------------
_FILE_CACHE: dict = {}
_ZERO_CV_CACHE: dict = {}


def _env(name, default=None):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v


def _mode():
    return (_env("FEDVLA_BASELINE", "") or "").strip().lower()


def _load_pt(path):
    """读 .pt（带缓存）。返回 None 表示文件不存在。"""
    if not path:
        return None
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]
    if not os.path.exists(path):
        print(f"[FedVLA-BASE] 找不到 {path} —— 该基线本轮无法生效（静默跳过）")
        _FILE_CACHE[path] = None
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = blob.get("cv", blob) if (isinstance(blob, dict) and "cv" in blob) else blob
    _FILE_CACHE[path] = out
    return out


def _named_params(model):
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            out[name] = p
    return out


def _import_fed_baselines():
    """定位 federated/fed_baselines.py 并 import（**不依赖 CWD** —— 铁律 F/G）。"""
    import sys
    _here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(_here, "..", "..", "federated"),
             os.path.join(_here, "..", "..", "..", "federated"),
             os.path.join(_here, "..", "federated")]
    for _p in cands:
        _p = os.path.abspath(_p)
        if os.path.exists(os.path.join(_p, "fed_baselines.py")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
            from fed_baselines import (apply_prox_term, scaffold_direction,
                                       scaffold_keys, scaffold_update_local_cv)
            return (apply_prox_term, scaffold_direction, scaffold_keys,
                    scaffold_update_local_cv)
    raise ImportError("找不到 federated/fed_baselines.py（请确认 federated/ 目录已同步到云端）")


def _zeros_like_keys(ref: dict, keys):
    """构造零控制变量（标准 SCAFFOLD 的零初始化）。带缓存。"""
    tag = tuple(sorted(keys))
    if tag not in _ZERO_CV_CACHE:
        _ZERO_CV_CACHE[tag] = {k: torch.zeros_like(ref[k]) for k in keys if k in ref}
    return _ZERO_CV_CACHE[tag]


def apply_baseline_hook(model, opt):
    """在 backward() 之后、optimizer.step() 之前修正梯度。

    FedProx : p.grad += mu * (p - w_global)
    SCAFFOLD: p.grad += (c_global - c_local)
    """
    mode = _mode()
    if mode not in ("fedprox", "scaffold"):
        return

    (apply_prox_term, scaffold_direction, scaffold_keys,
     _scaffold_update_local_cv) = _import_fed_baselines()

    np_ = _named_params(model)

    if mode == "fedprox":
        mu = float(_env("FEDVLA_PROX_MU", "0") or 0)
        if mu <= 0:
            return
        anchor = _load_pt(_env("FEDVLA_PROX_ANCHOR"))
        if anchor is None:
            return
        # 只对**模型里真实存在且锚点里也有**的参数生效（键空间必须一致）
        keys = [k for k in scaffold_keys(np_) if k in anchor]
        if not keys:
            print("[FedVLA-BASE] 锚点与模型参数键不匹配 —— 近端项未生效（静默退化！）")
            return
        n = apply_prox_term(list(np_.values()), np_, anchor, mu, keys)
        print(f"[FedVLA-BASE] FedProx 近端项已施加 mu={mu} 键数={len(keys)} "
              f"‖g_prox‖₂={n:.4e}")

    elif mode == "scaffold":
        c_g = _load_pt(_env("FEDVLA_SCAFFOLD_C"))
        c_l = _load_pt(_env("FEDVLA_SCAFFOLD_CV_LOCAL"))
        keys = [k for k in scaffold_keys(np_) if k in np_]
        if not c_g:
            c_g = _zeros_like_keys(np_, keys)      # 标准 SCAFFOLD：第 0 轮 c = 0
        if not c_l:
            c_l = _zeros_like_keys(np_, keys)      # 标准 SCAFFOLD：c_k 初值 0
        n = 0
        for name in keys:
            p = np_[name]
            if p.grad is None:
                continue
            g_c = c_g.get(name)
            c_li = c_l.get(name)
            if g_c is None and c_li is None:
                continue
            p.grad.copy_(scaffold_direction(p.grad, c_li, g_c))
            n += 1
        print(f"[FedVLA-BASE] SCAFFOLD 控制变量已应用 键数={n}")


def finish_baseline_round(model):
    """train() 结尾：SCAFFOLD 更新本地控制变量 c_k 并落盘。

    ⚠️ HPT 的 run.py 是**按 epoch 调 train()** 的（每轮 LOCAL_EPOCHS 次），
       所以本函数每轮会被调用多次；**最后一次**（epoch 末）的模型才是
       本地训练的最终结果 ⇒ 最后一次写出的 c_k 才是正确的。
    """
    mode = _mode()
    if mode != "scaffold":
        return

    out_path = _env("FEDVLA_SCAFFOLD_CV_OUT")
    if not out_path:
        return

    (_ap, _sd, scaffold_keys, scaffold_update_local_cv) = _import_fed_baselines()

    w_g = _load_pt(_env("FEDVLA_SCAFFOLD_WGLOBAL")) or {}
    c_g = _load_pt(_env("FEDVLA_SCAFFOLD_C")) or {}
    c_l = _load_pt(_env("FEDVLA_SCAFFOLD_CV_LOCAL")) or {}
    w_l = {name: p.detach().cpu() for name, p in model.named_parameters()}

    keys = [k for k in scaffold_keys(w_g) if k in w_g]
    if not keys:
        print("[FedVLA-BASE] SCAFFOLD w_global 键与模型不匹配 —— 跳过控制变量更新")
        return

    if not c_g:
        c_g = {k: torch.zeros_like(w_g[k]) for k in keys}
    if not c_l:
        c_l = {k: torch.zeros_like(w_g[k]) for k in keys}

    steps = int(float(_env("FEDVLA_SCAFFOLD_STEPS", "0") or 0))
    lr = float(_env("FEDVLA_SCAFFOLD_LR", "0") or 0)

    new_cv = scaffold_update_local_cv(c_l, c_g, w_g, w_l, steps, lr, keys)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save({"cv": new_cv}, out_path)
    print(f"[FedVLA-BASE] SCAFFOLD 本地控制变量已更新：键数={len(new_cv)} "
          f"K={steps} eta={lr} -> {out_path}")
