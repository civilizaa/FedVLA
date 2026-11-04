#!/usr/bin/env python3
"""FedVLA：把 MoE 补丁打到真实 HPT transformer 源码上做功能冒烟测试。

验证四件事：
  1) 打补丁后的 SimpleTransformer 能用 moe_cfg 构造出 MoE 层，且只在指定的层
  2) 前向能跑通，形状正确
  3) **训练起点与稠密模型函数等价**（载入预训练稠密 FFN 权重后输出完全一致）
  4) 专家计数能被收集并堆成 (L, E)

不需要 GPU、不需要 hydra/omegaconf（transformer.py 只依赖 torch + einops + timm，
timm 缺失时自动打桩）。

跑法：
    python federated/test_moe_transformer_smoke.py
"""
from __future__ import annotations

import os
import shutil
import sys
import types

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, cond, extra=""):
    if cond:
        print(f"  [OK] {name}")
    else:
        print(f"  [FAIL] {name} {extra}")
        FAILED.append(name)


def _stub_timm():
    """timm 缺失时打桩：DropPath 与 trunc_normal_ 的最小实现"""
    try:
        import timm.models.layers  # noqa: F401
        return
    except Exception:
        pass
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")

    class DropPath(torch.nn.Module):
        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep)
            return x * mask / keep

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        with torch.no_grad():
            return tensor.normal_(mean, std).clamp_(mean + a * std, mean + b * std)

    layers.DropPath = DropPath
    layers.trunc_normal_ = trunc_normal_
    models.layers = layers
    timm.models = models
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers


def build_patched_tree(tmpdir):
    """把本地缓存的 HPT 源码 + 两个补丁组装成一份可导入的 HPT 树"""
    ws = os.path.dirname(os.path.dirname(HERE))   # 工作区根目录（源码缓存在那里）
    srcs = {
        "transformer": os.path.join(ws, "hpt_transformer.py"),
        "policy": os.path.join(ws, "hpt_policy.py"),
        "run": os.path.join(ws, "hpt_run.py"),
    }
    for name, p in srcs.items():
        if not os.path.exists(p):
            raise SystemExit(f"[!!] 缺少 HPT 源码缓存 {p}；请先 curl 下载到工作区")
    hpt = os.path.join(tmpdir, "hpt")
    os.makedirs(os.path.join(hpt, "models"), exist_ok=True)
    os.makedirs(os.path.join(hpt, "utils"), exist_ok=True)
    shutil.copy(srcs["transformer"], os.path.join(hpt, "models", "transformer.py"))
    shutil.copy(srcs["policy"], os.path.join(hpt, "models", "policy.py"))
    shutil.copy(srcs["run"], os.path.join(hpt, "run.py"))
    open(os.path.join(hpt, "__init__.py"), "w").close()
    open(os.path.join(hpt, "models", "__init__.py"), "w").close()
    open(os.path.join(hpt, "utils", "__init__.py"), "w").close()
    return hpt


def _extract_fn(src: str, name: str) -> str:
    """从源码里截出某个顶层函数/类的完整定义文本（含其 docstring）。"""
    start = src.index(f"def {name}(")
    # 找到下一个顶层 def/class 作为结束
    rest = src[start + 10:]
    for marker in ("\ndef ", "\nclass "):
        idx = rest.find(marker)
        if idx != -1:
            end = start + 10 + idx + 1
            break
    else:
        end = len(src)
    return src[start:end]


def test_load_trunk_adapter(tmp, tr_moe, tr_dense, mod):
    """用**打好的补丁里的真实代码**测试 load_trunk 的 MoE 适配。

    为什么单独测：policy.py 依赖 hydra/omegaconf/normalizer 一大堆，本地装不全；
    但注入进去的 `_fedvla_load_trunk` 是个独立函数，直接把它抠出来 exec 就能测
    —— 测的正是云端会跑的那段代码，不是复制品。
    """
    import logging

    policy_path = os.path.join(tmp, "hpt", "models", "policy.py")
    src = open(policy_path, "r", encoding="utf-8").read()
    fn_src = _extract_fn(src, "_fedvla_load_trunk")
    ns = {"logging": logging}
    exec(compile(fn_src, policy_path, "exec"), ns)
    load_trunk = ns["_fedvla_load_trunk"]

    class FakePolicy:
        def __init__(self, trunk):
            self.trunk = trunk

    # 场景 A：MoE 模型 + 稠密预训练权重（第 0 轮从 hf:// 加载就是这个场景）
    dense_sd = tr_dense.state_dict()
    try:
        load_trunk(FakePolicy(tr_moe), dense_sd)
        check("[A] MoE 模型能载入稠密预训练权重（不报错）", True)
    except Exception as e:  # noqa: BLE001
        check("[A] MoE 模型能载入稠密预训练权重（不报错）", False,
              f"{type(e).__name__}: {e}")

    # 载入后 MoE 的输出投影仍是零 → 训练起点仍等于预训练模型
    with torch.no_grad():
        moe_max = 0.0
        for _n, m in tr_moe.named_modules():
            if isinstance(m, mod.TopKMoE):
                moe_max = max(moe_max, float(m.experts[0].fc2.weight.abs().max()))
    check("[A] 载入后 MoE 输出投影仍为 0", moe_max < 1e-9, f"{moe_max:.3e}")

    # 场景 B：非 MoE 模型 -> 仍然是 strict=True（多余键必须报错）
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
    try:
        load_trunk(FakePolicy(Tiny()), {"lin.weight": torch.zeros(2, 2),
                                         "lin.bias": torch.zeros(2),
                                         "bogus": torch.zeros(1)})
        check("[B] 非 MoE 模型的多余键被拒绝（strict=True 语义）", False, "居然没报错")
    except RuntimeError:
        check("[B] 非 MoE 模型的多余键被拒绝（strict=True 语义）", True)

    # 场景 C：MoE 模型缺了**非 MoE** 键 -> 必须报错（不能默默放过）
    bad = dict(dense_sd)
    victim = [k for k in bad if "blocks.0" in k]
    if victim:
        del bad[victim[0]]
        try:
            load_trunk(FakePolicy(tr_moe), bad)
            check("[C] MoE 模型缺非 MoE 键时报错", False, "居然没报错")
        except RuntimeError as e:
            check("[C] MoE 模型缺非 MoE 键时报错",
                  "缺失非 MoE 键" in str(e), str(e)[:100])

    # 场景 D：两个分支都必须走适配入口（hf 分支尤其重要 —— 第 0 轮就靠它）
    check("[D] hf:// 分支已改为走适配入口",
          "_fedvla_load_trunk(self, torch.load(path))" in src,
          "hf 分支仍是 strict=True，第 0 轮会因 MoE 新增键直接崩溃")
    check("[D] 本地分支已改为走适配入口",
          "_fedvla_load_trunk(self, sd)" in src)
    check("[D] 原版的两处 strict=True 调用都已替换",
          src.count("self.trunk.load_state_dict(") == 0,
          f"仍残留 {src.count('self.trunk.load_state_dict(')} 处")
    check("[D] 适配入口内部保留了 strict / strict=False 两条路径",
          fn_src.count("trunk.load_state_dict(") == 2,
          f"helper 内出现 {fn_src.count('trunk.load_state_dict(')} 次")


def main():
    import subprocess
    import tempfile

    _stub_timm()
    tmp = tempfile.mkdtemp(prefix="fedvla_moe_")
    try:
        hpt = build_patched_tree(tmp)
        py = sys.executable
        # 打补丁：顺序与云端一致 —— 先 load_trunk，再全局 lr，最后 MoE。
        # （MoE 补丁依赖 load_trunk 补丁；global_lr 与 MoE 都改 run.py，
        #   锚点必须不相交 —— 这一步就是在验证这一点）
        # 注意：patch_hpt_load_trunk.py 的 --target 指的是 **policy.py 文件**，
        #       其余两个补丁的 --hpt-dir 指的是 **HPT 根目录**
        for patcher, flag, target in (
                ("patch_hpt_load_trunk.py", "--target",
                 os.path.join(tmp, "hpt", "models", "policy.py")),
                ("patch_hpt_global_lr.py", "--hpt-dir", tmp),
                ("patch_hpt_moe.py", "--hpt-dir", tmp)):
            r = subprocess.run([py, os.path.join(HERE, patcher), flag, target],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[!!] {patcher} 打补丁失败")
                print(r.stdout[-2500:])
                print(r.stderr[-2500:])
                raise SystemExit(f"[!!] 打补丁失败：{patcher}")
            print(f"  [OK] {patcher} 已应用")

        # 两个补丁的标记都要在 run.py 里（锚点不相交的直接证据）
        run_src = open(os.path.join(tmp, "hpt", "run.py"), encoding="utf-8").read()
        check("run.py 同时含全局 lr 调度与 MoE 导出两段注入",
              "GlobalStepLRScheduler" in run_src and "FedVLA-MOE" in run_src)
        check("MoE 导出代码没有挤掉 global_lr 的 optimizer 状态保存",
              "_opt_state" in run_src and "FEDVLA_MOE_COUNTS" in run_src)
        check("两段注入代码都在 run() 里（缩进为 4 空格）",
              "\n    # [FedVLA patch] 保存 optimizer 状态" in run_src.replace("\r\n", "\n")
              and "\n    # [FedVLA-MOE] 导出专家使用计数" in run_src.replace("\r\n", "\n"))
        check("'model saved to' 打印仍在（没被任一补丁吃掉）",
              'print("model saved to :", policy_path)' in run_src)

        sys.path.insert(0, tmp)
        # 让 hpt.utils.fedvla_moe 可导入
        import importlib
        mod = importlib.import_module("hpt.utils.fedvla_moe")
        transformer = importlib.import_module("hpt.models.transformer")

        print("\n[1] 构造带 MoE 的 SimpleTransformer")
        from functools import partial
        torch.manual_seed(0)
        dim, nb = 32, 4
        attn = partial(transformer.MultiheadAttention, embed_dim=dim,
                       num_heads=4, bias=True, add_bias_kv=True)
        moe_cfg = {"enabled": True, "num_experts": 8, "top_k": 2,
                   "expert_ratio": 0.25, "layers": "half"}
        tr_moe = transformer.SimpleTransformer(
            attn_target=attn, embed_dim=dim, num_blocks=nb,
            pre_transformer_layer=None, post_transformer_layer=None, moe_cfg=moe_cfg)
        check("moe 层号 = 后一半", tr_moe.moe_layer_ids == [2, 3], f"{tr_moe.moe_layer_ids}")
        n_moe = sum(1 for m in tr_moe.modules() if isinstance(m, mod.TopKMoE))
        check("生成了 2 个 MoE 层", n_moe == 2, f"{n_moe}")

        print("\n[2] 前向")
        x = torch.randn(2, 5, dim)
        tr_moe.eval()
        with torch.no_grad():
            y = tr_moe(x)
        check("输出形状正确", tuple(y.shape) == (2, 5, dim), f"{tuple(y.shape)}")

        print("\n[3] 与稠密模型的函数等价（载入预训练权重后）")
        tr_dense = transformer.SimpleTransformer(
            attn_target=attn, embed_dim=dim, num_blocks=nb,
            pre_transformer_layer=None, post_transformer_layer=None)
        # 稠密模型的 state_dict 直接载入 MoE 模型（strict=False）：
        # 稠密 FFN 的键名完全一致，不需要任何形状变换。
        dense_sd = tr_dense.state_dict()
        missing, unexpected = tr_moe.load_state_dict(dense_sd, strict=False)
        allowed = mod.moe_param_keys(tr_moe)
        check("缺失键全部属于 MoE 分支",
              set(missing) == allowed and len(missing) > 0,
              f"{sorted(missing)[:3]} ...")
        check("无多余键", len(unexpected) == 0, f"{list(unexpected)[:4]}")
        tr_moe.eval()
        tr_dense.eval()
        with torch.no_grad():
            y_moe = tr_moe(x)
            y_dense = tr_dense(x)
        err = (y_moe - y_dense).abs().max().item()
        check("输出与稠密模型一致", err < 1e-4, f"max err={err:.3e}")

        print("\n[3b] 构造后零初始化未被 _init_weights 覆盖")
        # HPT 的 SimpleTransformer 在构造末尾会 apply(_init_weights)，
        # 那一步会覆盖 MoE 分支输出投影的零值 —— 补丁必须在之后补回来。
        n_zeroed = mod.zero_moe_output_projections(tr_moe)
        check("能识别出 2 个 MoE 分支", n_zeroed == 2, f"{n_zeroed}")
        with torch.no_grad():
            moe_out_max = 0.0
            for name, m in tr_moe.named_modules():
                if isinstance(m, mod.TopKMoE):
                    moe_out_max = max(moe_out_max,
                                      float(m(x).abs().max()))
        check("MoE 分支输出为 0", moe_out_max < 1e-6, f"{moe_out_max:.3e}")

        print("\n[4] 专家计数")
        tr_moe.train()
        tr_moe(x)
        counts = mod.collect_expert_counts(tr_moe)
        check("收集到 2 层", len(counts) == 2, f"{list(counts)}")
        M = mod.stack_expert_counts(counts)
        check("堆成 (L, E) = (2, 8)", tuple(M.shape) == (2, 8), f"{tuple(M.shape)}")
        # 计数是"每层独立"的：总 token 数 = 层数 × B × T，每 token 记 k 次
        expect = M.shape[0] * 2 * 5 * 2
        check("计数总和 = 层数×token×k", abs(M.sum().item() - expect) < 1e-4,
              f"{M.sum().item()} vs {expect}")
        check("每层计数总和 = token×k = 20",
              abs(M[0].sum().item() - 20) < 1e-4, f"{M[0].sum().item()}")

        print("\n[5] 关掉 MoE 时行为与原版一致")
        tr_plain = transformer.SimpleTransformer(
            attn_target=attn, embed_dim=dim, num_blocks=nb,
            pre_transformer_layer=None, post_transformer_layer=None)
        check("无 MoE 层", sum(1 for m in tr_plain.modules()
                               if isinstance(m, mod.TopKMoE)) == 0)
        check("moe_layer_ids 为空", tr_plain.moe_layer_ids == [])

        print("\n[6] load_trunk 的 MoE 适配（真实补丁代码）")
        test_load_trunk_adapter(tmp, tr_moe, tr_dense, mod)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"!! {len(FAILED)} 项失败: {FAILED}")
        return 1
    print("全部通过  ALL_TESTS_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
