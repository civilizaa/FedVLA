"""给 HPT 打「MoE + 专家计数导出」补丁（幂等）：给联邦基线装上标准 top-k MoE。

═══════════════════════════════════════════════════════════════════════════════
背景
═══════════════════════════════════════════════════════════════════════════════
第 3 阶段要在 HPT-Base 的 trunk 里插 MoE，用来：
  1) 给 EDA（专家驱动聚合）提供「专家选择计数」信号；
  2) 让不同任务的数据自然路由到不同专家，从而在参数层面实现任务分化。

设计上刻意避开 FedVLA 原文的**双门控**（`W_e` + `lam`），因为它的初始化尺度不对
时会出现「要么全激活、要么几乎不激活」的二值化崩塌（已在 fedvla_scaffold 的 toy
实验里观察到）。这里只用**标准 top-k 路由**（E=8, k=2），并采用
「稠密 FFN + 并联零初始化 MoE 残差」的结构 —— 训练起点**严格等于预训练模型**，
预训练权重**原样载入**（不需要任何键名/形状变换）。
详见 federated/fedvla_moe.py 的模块文档。

═══════════════════════════════════════════════════════════════════════════════
补丁内容
═══════════════════════════════════════════════════════════════════════════════
1) 复制 federated/fedvla_moe.py -> hpt/utils/fedvla_moe.py
2) hpt/models/transformer.py
     - BlockWithMasking.__init__ 增加 moe_cfg 参数，用 MoEBlock 替代 Mlp
     - SimpleTransformer.__init__ 增加 moe_cfg 参数，按层号决定哪些 block 用 MoE
     - 构造末尾补一次零初始化（_init_weights 会覆盖构造时的零值）
3) hpt/models/policy.py
     - _create_policy_trunk 从环境变量读取 MoE 配置并传给 SimpleTransformer
     - load_trunk（本地分支）做 MoE 适配：允许"MoE 新增键"缺失，其余仍严格校验
4) hpt/run.py
     - 训练结束后把专家计数导出到 FEDVLA_MOE_COUNTS 指定的文件

环境变量（全部可选，默认不改变 HPT 行为）：
  FEDVLA_MOE=1              开启 MoE
  FEDVLA_MOE_EXPERTS=8      专家数
  FEDVLA_MOE_TOPK=2         每 token 激活的专家数
  FEDVLA_MOE_EXPERT_RATIO=0.25   每个专家的 hidden dim 相对稠密 FFN 的比例
  FEDVLA_MOE_LAYERS=half    half(后一半) | all | 逗号分隔的层号
  FEDVLA_MOE_COUNTS=<path>  专家计数导出路径

用法（在 AutoDL 上）：
    python /root/HPT/federated/patch_hpt_moe.py             # 打补丁
    python /root/HPT/federated/patch_hpt_moe.py --verify    # 只检查
    python /root/HPT/federated/patch_hpt_moe.py --revert    # 回滚
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

DEFAULT_HPT_DIR = os.path.join(os.path.expanduser("~"), "HPT")
MARK = "[FedVLA-MOE]"


# ══════════════════════════════════════════════════════════════════════════
# 1) transformer.py
# ══════════════════════════════════════════════════════════════════════════
T_SIG_ANCHOR = """        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
    ):
        super().__init__()
"""
T_SIG_NEW = """        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 1e-4,
        moe_cfg: Optional[dict] = None,   # [FedVLA-MOE] top-k MoE 配置
    ):
        super().__init__()
"""

T_MLP_ANCHOR = """        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
"""
T_MLP_NEW = """        # [FedVLA-MOE] 用「稠密 FFN + 并联零初始化 MoE 残差」替代纯 Mlp
        # （moe_cfg 为空时行为与原版完全一致）
        _moe = moe_cfg if isinstance(moe_cfg, dict) and moe_cfg.get("enabled") else None
        if _moe is not None:
            from hpt.utils.fedvla_moe import MoEBlock
            self.mlp = MoEBlock(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                num_experts=int(_moe.get("num_experts", 8)),
                top_k=int(_moe.get("top_k", 2)),
                act_layer=act_layer,
                drop=ffn_dropout_rate,
                expert_ratio=float(_moe.get("expert_ratio", 0.25)),
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=ffn_dropout_rate,
            )
"""

T_ST_SIG_ANCHOR = """        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "pytorch",  # possible values jax or pytorch
    ):
"""
T_ST_SIG_NEW = """        layer_scale_init_value: float = 1e-4,  # from cait; float
        weight_init_style: str = "pytorch",  # possible values jax or pytorch
        moe_cfg: Optional[dict] = None,  # [FedVLA-MOE] top-k MoE 配置
    ):
"""

T_BLOCKS_ANCHOR = """        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
"""
T_BLOCKS_NEW = """        # [FedVLA-MOE] 决定哪些 block 用 MoE
        _moe_layer_ids = set()
        if isinstance(moe_cfg, dict) and moe_cfg.get("enabled"):
            _spec = str(moe_cfg.get("layers", "half"))
            if _spec == "all":
                _moe_layer_ids = set(range(num_blocks))
            elif _spec == "half":
                _moe_layer_ids = set(range(num_blocks // 2, num_blocks))
            else:
                _moe_layer_ids = {int(x) for x in _spec.split(",") if x.strip().isdigit()}
        self.moe_layer_ids = sorted(_moe_layer_ids)
        self.blocks = nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                    **({"moe_cfg": dict(moe_cfg)} if (i in _moe_layer_ids and block is BlockWithMasking) else {}),
                )
                for i in range(num_blocks)
            ]
        )
"""

T_INIT_ANCHOR = """        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)
"""
T_INIT_NEW = """        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)
        # [FedVLA-MOE] _init_weights 会覆盖 MoE 分支输出投影的零初始化，这里补回来，
        # 保证「训练起点严格等于预训练模型」。
        try:
            from hpt.utils.fedvla_moe import zero_moe_output_projections
            _n_moe = zero_moe_output_projections(self)
            if _n_moe:
                print(f"[FedVLA-MOE] zero-init {_n_moe} 个 MoE 分支的输出投影")
        except Exception:
            pass
"""


# ══════════════════════════════════════════════════════════════════════════
# 2) policy.py
# ══════════════════════════════════════════════════════════════════════════
P_TRUNK_ANCHOR = """                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
            )
"""
P_TRUNK_NEW = """                post_transformer_layer=EinOpsRearrange("l b d -> b l d"),
                weight_init_style=weight_init_style,
                # [FedVLA-MOE] 从环境变量读取 MoE 配置（不设则完全保持原版行为）
                moe_cfg=(
                    {
                        "enabled": True,
                        "num_experts": int(os.environ.get("FEDVLA_MOE_EXPERTS", "8")),
                        "top_k": int(os.environ.get("FEDVLA_MOE_TOPK", "2")),
                        "expert_ratio": float(os.environ.get("FEDVLA_MOE_EXPERT_RATIO", "0.25")),
                        "layers": os.environ.get("FEDVLA_MOE_LAYERS", "half"),
                    }
                    if os.environ.get("FEDVLA_MOE", "0") == "1" else None
                ),
            )
"""

# load_trunk 的 **hf 分支**（第 0 轮从 hf://liruiw/hpt-base 加载预训练权重时走这里）
# ⚠️ 这一支同样用 strict=True —— 不打补丁的话，MoE 新增键会被判成 missing 而直接报错，
#    第 0 轮根本起不来。
P_LOAD_HF_ANCHOR = """            self.trunk.load_state_dict(torch.load(path), strict=True)
"""
P_LOAD_HF_NEW = """            # [FedVLA-MOE] 走统一的适配入口（允许 MoE 新增键缺失）
            _fedvla_load_trunk(self, torch.load(path))
"""

# load_trunk 的本地分支（必须匹配 patch_hpt_load_trunk.py 打过的版本）
P_LOAD_ANCHOR = """            self.trunk.load_state_dict(sd, strict=True)
"""
P_LOAD_NEW = """            # [FedVLA-MOE] 走统一的适配入口（允许 MoE 新增键缺失）
            _fedvla_load_trunk(self, sd)
"""

# 注入到 policy.py 模块级的辅助函数
P_HELPER_ANCHOR = """class Policy(nn.Module):
"""
P_HELPER_NEW = '''# [FedVLA-MOE] 统一的 trunk 载入入口
def _fedvla_load_trunk(policy, sd):
    """载入 trunk 权重。

    有 MoE 时：稠密 FFN 的权重原样载入（键名不变），只允许"MoE 新增键"缺失
    （它们保持构造时的初始化 —— 输出投影是零，所以训练起点严格等于预训练模型）；
    其余键仍然严格校验（多了/少了非 MoE 键一律报错）。
    没有 MoE 时：完全等价于原来的 strict=True。
    """
    _has_moe = False
    try:
        from hpt.utils.fedvla_moe import has_moe, moe_param_keys
        _has_moe = has_moe(policy.trunk)
    except Exception:
        _has_moe = False
    if not _has_moe:
        policy.trunk.load_state_dict(sd, strict=True)
        return
    _allowed = moe_param_keys(policy.trunk)
    _missing, _unexpected = policy.trunk.load_state_dict(sd, strict=False)
    if _unexpected:
        raise RuntimeError(
            f"[FedVLA-MOE] trunk 载入出现多余键（前 5 个）: {list(_unexpected)[:5]}")
    _bad = [k for k in _missing if k not in _allowed]
    if _bad:
        raise RuntimeError(
            f"[FedVLA-MOE] trunk 载入缺失非 MoE 键（前 5 个）: {list(_bad)[:5]}")
    logging.info(f"[FedVLA-MOE] MoE trunk 载入完成，"
                 f"{len(_missing)} 个 MoE 新增参数保持初始值")


class Policy(nn.Module):
'''


# ══════════════════════════════════════════════════════════════════════════
# 3) run.py —— 训练结束后导出专家计数
# ══════════════════════════════════════════════════════════════════════════
# ⚠️ 锚点必须与 patch_hpt_global_lr.py 的 ANCHOR2 **完全不相交**：
#    global_lr 用的是 "    # log and finish\n    print(\"model saved to :\", policy_path)\n"，
#    两个补丁都用 replace(anchor, new, 1) —— 共用锚点的话，先打的会把锚点吃掉，
#    后打的必然报「找不到锚点」，云端在打补丁阶段就挂。
#    这里改用 global_lr 完全不会碰的 "pbar.close()/run.finish()/wandb.finish()" 三行。
R_ANCHOR = """    pbar.close()
    run.finish()
    wandb.finish()
"""
R_NEW = '''    # [FedVLA-MOE] 导出专家使用计数（EDA 聚合的输入信号）
    _moe_counts_path = os.environ.get("FEDVLA_MOE_COUNTS", "")
    if _moe_counts_path:
        try:
            from hpt.utils.fedvla_moe import collect_expert_counts, stack_expert_counts
            _counts = collect_expert_counts(policy)
            _M = stack_expert_counts(_counts)
            torch.save({"counts": _M, "names": sorted(_counts.keys())}, _moe_counts_path)
            print(f"[FedVLA-MOE] expert counts saved to {_moe_counts_path} shape={tuple(_M.shape)}")
        except Exception as _e:
            print(f"[FedVLA-MOE] expert counts export FAILED ({type(_e).__name__}: {_e})")

    pbar.close()
    run.finish()
    wandb.finish()
'''


def find_hpt_dir(explicit):
    if explicit:
        return explicit
    for cand in (DEFAULT_HPT_DIR, "/root/HPT", os.getcwd()):
        if os.path.isfile(os.path.join(cand, "hpt", "run.py")):
            return cand
    raise FileNotFoundError("找不到 HPT 源码目录（需要含 hpt/run.py），请用 --hpt-dir 指定")


def _apply(src, anchor, new, what, path):
    if anchor not in src:
        raise SystemExit(f"[!!] 找不到锚点（{what}），HPT 版本可能不同，请手工检查：\n     {path}")
    return src.replace(anchor, new, 1)


def _moe_source():
    """从同目录的 federated/fedvla_moe.py 取模块源码（单一真源，避免两处不同步）"""
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "fedvla_moe.py"),
                 "/root/HPT/federated/fedvla_moe.py"):
        if os.path.exists(cand):
            return open(cand, "r", encoding="utf-8").read(), cand
    raise SystemExit("[!!] 找不到 fedvla_moe.py（应与本脚本同目录）")


def main(argv=None):
    ap = argparse.ArgumentParser(description="给 HPT 打 MoE + 专家计数导出补丁")
    ap.add_argument("--hpt-dir", default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args(argv)

    hpt_dir = find_hpt_dir(args.hpt_dir)
    files = {
        "transformer": os.path.join(hpt_dir, "hpt", "models", "transformer.py"),
        "policy": os.path.join(hpt_dir, "hpt", "models", "policy.py"),
        "run": os.path.join(hpt_dir, "hpt", "run.py"),
    }
    moe_py = os.path.join(hpt_dir, "hpt", "utils", "fedvla_moe.py")

    if args.revert:
        ok = True
        for name, path in files.items():
            bak = path + ".fedvla_moe.bak"
            if os.path.exists(bak):
                shutil.copy(bak, path)
                print(f"[OK] 已回滚 {name}: {bak} -> {path}")
            else:
                print(f"[SKIP] {name} 无备份")
                ok = False
        print(f"     提示：{moe_py} 保留无害（不被引用）")
        return 0 if ok else 2

    srcs = {k: open(v, "r", encoding="utf-8").read() for k, v in files.items()}

    if args.verify:
        ok = os.path.exists(moe_py)
        for name, src in srcs.items():
            has = MARK in src
            print(f"  {name:<12} 已打补丁: {has}")
            ok = ok and has
        print(f"  fedvla_moe.py 存在: {os.path.exists(moe_py)}")
        print("[OK] 补丁完整" if ok else "[!!] 补丁不完整")
        return 0 if ok else 1

    if all(MARK in s for s in srcs.values()):
        print("[OK] 补丁已存在，无需重复打（幂等）")
        if not os.path.exists(moe_py):
            src, srcpath = _moe_source()
            os.makedirs(os.path.dirname(moe_py), exist_ok=True)
            with open(moe_py, "w", encoding="utf-8") as f:
                f.write(src)
            print(f"[OK] 补写模块：{moe_py}（源 {srcpath}）")
        return 0

    # 写模块文件（单一真源）
    src, srcpath = _moe_source()
    os.makedirs(os.path.dirname(moe_py), exist_ok=True)
    with open(moe_py, "w", encoding="utf-8") as f:
        f.write(src)
    print(f"[OK] 已写入 {moe_py}（源 {srcpath}）")

    # ---- transformer.py ----
    t = srcs["transformer"]
    t = _apply(t, T_SIG_ANCHOR, T_SIG_NEW, "BlockWithMasking.__init__ 签名", files["transformer"])
    t = _apply(t, T_MLP_ANCHOR, T_MLP_NEW, "BlockWithMasking 的 mlp 构造", files["transformer"])
    t = _apply(t, T_ST_SIG_ANCHOR, T_ST_SIG_NEW, "SimpleTransformer.__init__ 签名", files["transformer"])
    t = _apply(t, T_BLOCKS_ANCHOR, T_BLOCKS_NEW, "SimpleTransformer 的 blocks 构造", files["transformer"])
    t = _apply(t, T_INIT_ANCHOR, T_INIT_NEW, "SimpleTransformer 的 _init_weights", files["transformer"])

    # ---- policy.py ----
    p = srcs["policy"]
    p = _apply(p, P_HELPER_ANCHOR, P_HELPER_NEW, "policy.py 的 class Policy 之前", files["policy"])
    p = _apply(p, P_TRUNK_ANCHOR, P_TRUNK_NEW, "_create_policy_trunk 的 SimpleTransformer 调用", files["policy"])
    p = _apply(p, P_LOAD_HF_ANCHOR, P_LOAD_HF_NEW, "load_trunk 的 hf:// 分支", files["policy"])
    p = _apply(p, P_LOAD_ANCHOR, P_LOAD_NEW, "load_trunk 的本地分支", files["policy"])

    # ---- run.py ----
    r = srcs["run"]
    r = _apply(r, R_ANCHOR, R_NEW, "训练结束的保存段", files["run"])

    # 备份并写回
    for name, path in files.items():
        bak = path + ".fedvla_moe.bak"
        if not os.path.exists(bak):
            shutil.copy(path, bak)
            print(f"[OK] 已备份 {name} -> {bak}")
    with open(files["transformer"], "w", encoding="utf-8") as f:
        f.write(t)
    with open(files["policy"], "w", encoding="utf-8") as f:
        f.write(p)
    with open(files["run"], "w", encoding="utf-8") as f:
        f.write(r)

    print("[OK] 已打补丁：")
    print("     1) hpt/utils/fedvla_moe.py: MoEBlock（稠密 FFN + 零初始化 MoE 残差）")
    print("     2) hpt/models/transformer.py: BlockWithMasking / SimpleTransformer 支持 moe_cfg")
    print("     3) hpt/models/policy.py: _create_policy_trunk 读环境变量；load_trunk 允许 MoE 新增键")
    print("     4) hpt/run.py: 训练结束导出 expert_counts.pt")
    print()
    print("下一步（云端）：")
    print("  export FEDVLA_MOE=1 FEDVLA_MOE_EXPERTS=8 FEDVLA_MOE_TOPK=2")
    print("  export FEDVLA_MOE_LAYERS=half FEDVLA_MOE_EXPERT_RATIO=0.25")
    print("  export FEDVLA_MOE_COUNTS=<client_out>/expert_counts.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
