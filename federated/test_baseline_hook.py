"""联邦基线钩子（fedvla_baseline_hook.py）的集成单测。

═══════════════════════════════════════════════════════════════════════════════
为什么单独测这一层
═══════════════════════════════════════════════════════════════════════════════
`test_fed_baselines.py` 验的是**数学**；本文件验的是**接线**——
也就是"钩子能不能真的把修正量作用到 HPT 模型的参数上"。

这里最关键的一条：**键空间**。
  * HPT 模型参数名   : `trunk.trunk.xxx`
  * 聚合写出的 trunk.pth 的键: `trunk.xxx`
两者不一致 ⇒ 若直接拿 trunk.pth 当锚点，修正项作用不到任何参数
⇒ **基线静默退化成 FedAvg，零报错**（第 4 阶段踩过的 silent failure 套路）。

⇒ 本文件专门验证：
  1. 键空间**一致**时，梯度确实被修正（近端项数值精确可验）；
  2. 键空间**不一致**时，钩子**拒绝生效并打印告警**（而不是静默 no-op）；
  3. SCAFFOLD 的本地控制变量按公式落盘，且键空间正确；
  4. 基线关闭时钩子是彻底的 no-op。

跑法：
    python federated/test_baseline_hook.py
"""
from __future__ import annotations

import importlib.util
import io
import os
import re
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 以"云端落盘后的样子"加载钩子模块（它会被复制到 hpt/utils/）
_spec = importlib.util.spec_from_file_location(
    "fedvla_baseline", os.path.join(HERE, "fedvla_baseline_hook.py"))
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


class FakeModel(torch.nn.Module):
    """模拟 HPT Policy：参数名形如 `trunk.trunk.xxx`（ModuleDict 嵌套的键空间）。

    ⚠️ 不用覆写 named_parameters —— torch 默认实现就给出
       `trunk.trunk.weight` / `stems.d_rgb.weight` 这类名字，
       正好复现真实 HPT 的键空间（Policy.trunk 是 ModuleDict）。
    """

    def __init__(self, dim=4):
        super().__init__()
        self.trunk = torch.nn.ModuleDict({"trunk": torch.nn.Linear(dim, dim)})
        self.stems = torch.nn.ModuleDict({"d_rgb": torch.nn.Linear(dim, dim)})


def _reset_env():
    for k in list(os.environ):
        if k.startswith("FEDVLA_BASELINE") or k.startswith("FEDVLA_PROX") \
                or k.startswith("FEDVLA_SCAFFOLD"):
            del os.environ[k]
    hook._FILE_CACHE.clear()
    hook._ZERO_CV_CACHE.clear()


def _tmp(name):
    return os.path.join(HERE, "_tmp_test", name)


def test_prox_hook_modifies_gradient():
    """【接线】键空间一致 ⇒ 钩子真的把 μ(w−w_g) 加到了 .grad 上。"""
    _reset_env()
    os.makedirs(_tmp(""), exist_ok=True)
    model = FakeModel()
    anchor = {n: torch.randn_like(p) for n, p in model.named_parameters()
              if n.startswith("trunk.")}
    ap = _tmp("anchor.pt")
    torch.save(anchor, ap)

    os.environ["FEDVLA_BASELINE"] = "fedprox"
    os.environ["FEDVLA_PROX_MU"] = "0.3"
    os.environ["FEDVLA_PROX_ANCHOR"] = ap

    # 造一个非零梯度
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = {n: p.grad.clone() for n, p in model.named_parameters()}

    hook.apply_baseline_hook(model, None)

    for n, p in model.named_parameters():
        if n.startswith("trunk."):
            expect = before[n] + 0.3 * (p.detach() - anchor[n])
            assert torch.allclose(p.grad, expect, atol=1e-6), f"{n} 近端项未生效"
        else:
            # 个性化参数（stems.*）**不能**被近端项改动
            assert torch.allclose(p.grad, before[n]), f"{n} 个性化参数被误改"
    print("  ✅ 钩子按预期修正了 trunk 梯度，且未触碰 stems")


def test_prox_hook_rejects_key_mismatch():
    """【防静默失败】键空间不一致 ⇒ 拒绝生效（而不是静默 no-op）。"""
    _reset_env()
    model = FakeModel()
    # 故意用 trunk.pth 的键空间（`trunk.xxx`），与模型参数名（`trunk.trunk.xxx`）不符
    bad = {"trunk.weight": torch.randn(4, 4), "trunk.bias": torch.randn(4)}
    bp = _tmp("bad_anchor.pt")
    torch.save(bad, bp)

    os.environ["FEDVLA_BASELINE"] = "fedprox"
    os.environ["FEDVLA_PROX_MU"] = "0.3"
    os.environ["FEDVLA_PROX_ANCHOR"] = bp

    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = {n: p.grad.clone() for n, p in model.named_parameters()}

    import io as _io
    import contextlib
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        hook.apply_baseline_hook(model, None)
    out = buf.getvalue()

    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, before[n]), f"{n} 键不匹配时不应修改梯度"
    assert "键不匹配" in out, f"应打印键空间告警，实际输出：{out}"
    print("  ✅ 键空间错配被拦下并告警（不会静默退化成 FedAvg）")


def test_scaffold_hook_applies_and_exports():
    """【接线】SCAFFOLD：梯度方向被修正 + 本地控制变量按公式落盘。"""
    _reset_env()
    model = FakeModel()
    keys = [n for n, _ in model.named_parameters() if n.startswith("trunk.")]

    c_g = {k: torch.full_like(model.state_dict()[k], 0.5) for k in keys}
    c_l = {k: torch.full_like(model.state_dict()[k], 0.2) for k in keys}
    w_g = {k: torch.full_like(model.state_dict()[k], 1.0) for k in keys}
    torch.save({"cv": c_g}, _tmp("c_global.pt"))
    torch.save({"cv": c_l}, _tmp("c_local.pt"))
    torch.save(w_g, _tmp("w_global.pt"))

    os.environ["FEDVLA_BASELINE"] = "scaffold"
    os.environ["FEDVLA_SCAFFOLD_C"] = _tmp("c_global.pt")
    os.environ["FEDVLA_SCAFFOLD_CV_LOCAL"] = _tmp("c_local.pt")
    os.environ["FEDVLA_SCAFFOLD_WGLOBAL"] = _tmp("w_global.pt")
    os.environ["FEDVLA_SCAFFOLD_CV_OUT"] = _tmp("cv_out.pt")
    os.environ["FEDVLA_SCAFFOLD_STEPS"] = "10"
    os.environ["FEDVLA_SCAFFOLD_LR"] = "0.1"

    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = {n: p.grad.clone() for n, p in model.named_parameters()}
    hook.apply_baseline_hook(model, None)
    for n, p in model.named_parameters():
        if n.startswith("trunk."):
            # g ← g − c_k + c = 1 − 0.2 + 0.5 = 1.3
            assert torch.allclose(p.grad, torch.full_like(p.grad, 1.3), atol=1e-6), \
                f"{n} SCAFFOLD 方向修正不对"
        else:
            assert torch.allclose(p.grad, before[n]), f"{n} 个性化参数被误改"

    # 导出本地控制变量
    hook.finish_baseline_round(model)
    assert os.path.exists(_tmp("cv_out.pt")), "本地控制变量未落盘"
    blob = torch.load(_tmp("cv_out.pt"), map_location="cpu", weights_only=False)
    cv = blob["cv"]
    assert set(cv.keys()) == set(keys), f"控制变量键空间不对: {list(cv.keys())}"
    # c_k⁺ = c_k − c + (w_g − w_k)/(K·η) = 0.2 − 0.5 + (1 − w_k)/1.0
    for k in keys:
        w_local = dict(model.named_parameters())[k].detach()
        expect = (c_l[k].float() - c_g[k].float()
                  + (w_g[k].float() - w_local.float()) / (10 * 0.1))
        assert torch.allclose(cv[k].float(), expect, atol=1e-5), f"{k} 控制变量更新不对"
    print("  ✅ SCAFFOLD 梯度修正 + 控制变量落盘均正确（键空间与公式都对）")


def test_hook_disabled_is_noop():
    """【零影响】BASELINE 未设置 ⇒ 钩子彻底 no-op。"""
    _reset_env()
    out_cv = _tmp("cv_out_noop.pt")
    if os.path.exists(out_cv):
        os.remove(out_cv)
    os.environ["FEDVLA_SCAFFOLD_CV_OUT"] = out_cv
    model = FakeModel()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = {n: p.grad.clone() for n, p in model.named_parameters()}
    hook.apply_baseline_hook(model, None)
    hook.finish_baseline_round(model)
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, before[n]), f"{n} 关闭时梯度被改了"
    assert not os.path.exists(out_cv), "关闭时不应产出控制变量"
    print("  ✅ 基线关闭时为彻底 no-op")


def test_key_space_conversion_helper():
    """【键空间】trunk.pth 的键 `trunk.xxx` 必须能转成全模型键空间 `trunk.trunk.xxx`。

    ⚠️ 这里的转换规则必须与 run_fedavg.sh 里 `_write_anchor` 的**完全一致**，
       否则单测通过但云端仍会键不匹配 ⇒ 基线静默失效。
       为避免两处逻辑漂移，本用例**直接执行 run_fedavg.sh 里的那段脚本**。
    """
    _reset_env()
    import subprocess
    import tempfile
    run_sh = os.path.join(HERE, "run_fedavg.sh")
    with open(run_sh, encoding="utf-8") as f:
        sh = f.read()
    # 抽出 _write_anchor 里的 python 片段
    m = re.search(r"_write_anchor \(\) \{.*?<<'PYANCHOR'\n(.*?)\nPYANCHOR\n", sh, re.S)
    assert m, "run_fedavg.sh 里找不到 _write_anchor 的 python 片段"
    snippet = m.group(1)

    with tempfile.TemporaryDirectory() as td:
        srcdir = os.path.join(td, "g")
        os.makedirs(srcdir)
        trunk = {"trunk.weight": torch.randn(4, 4), "trunk.bias": torch.randn(4)}
        torch.save(trunk, os.path.join(srcdir, "trunk.pth"))
        out = os.path.join(td, "anchor.pt")
        code = snippet.replace('"$HPT_PY" - "$1" "$2"', "sys.argv[0], sys.argv[1], sys.argv[2]")
        code = code.replace('src, dst = sys.argv[1], sys.argv[2]',
                            'src, dst = sys.argv[1], sys.argv[2]')
        py = os.path.join(td, "conv.py")
        with open(py, "w", encoding="utf-8") as f:
            f.write(code)
        r = subprocess.run([sys.executable, py, srcdir, out],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"转换脚本失败：{r.stderr}"
        full = torch.load(out, map_location="cpu", weights_only=False)

    model = FakeModel()
    model_names = {n for n, _ in model.named_parameters()}
    assert set(full.keys()) & model_names, \
        f"转换后仍对不上模型参数名：{list(full)} vs {sorted(model_names)}"
    print("  ✅ 键空间转换规则正确（与 run_fedavg.sh 同一份实现）")


def main():
    tests = [
        test_prox_hook_modifies_gradient,
        test_prox_hook_rejects_key_mismatch,
        test_scaffold_hook_applies_and_exports,
        test_hook_disabled_is_noop,
        test_key_space_conversion_helper,
    ]
    print("=" * 70)
    print("联邦基线钩子 集成单测（接线层，不需 GPU / 完整 HPT）")
    print("=" * 70)
    failed = []
    for t in tests:
        print(f"\n[{t.__name__}]")
        try:
            t()
        except AssertionError as e:
            print(f"  ❌ 失败：{e}")
            failed.append(t.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ 异常：{type(e).__name__}: {e}")
            failed.append(t.__name__)
    print("\n" + "=" * 70)
    if failed:
        print(f"❌ {len(failed)}/{len(tests)} 个用例失败：{failed}")
        sys.exit(1)
    print(f"✅ 全部 {len(tests)} 个用例通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
