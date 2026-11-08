#!/usr/bin/env python3
"""探针 stage4：视觉通路反演到底能不能成？——排除"优化不足"的可能。

要区分两种解释：
  (a) 维度诅咒：潜变量 25088 维 vs 梯度 17.1M 维 —— 视觉内容被高维淹没
  (b) 优化不足：步数/学习率/匹配集没调好 —— 再多算力就能反演出来

跑 4 组对照：
  V1  dense 匹配, 600 步, lr 0.05
  V2  dense 匹配, 300 步, lr 0.01
  V3  all   匹配, 300 步, lr 0.05
  V4  dense 匹配, 300 步, lr 0.05, **只反演 state**（低维对照）
外加：
  A1  action, dense, 300 步, lr 0.05（已知能成，做基准）
  S1  state,  dense, 300 步, lr 0.05（低维对照，验证维度假说）
"""
import io
import os
import sys
import time
import traceback

sys.path.insert(0, "/root/HPT")
sys.path.insert(0, "/root/HPT/federated")

import numpy as np
import torch

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

from privacy_experiment import (build_policy, compute_local_grad, make_batch,
                                split_uplink)
from inversion import invert_observation

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
    dev = torch.device("cuda")
    pol, cfg, d = build_policy("/root/HPT/output/p3_moe_eda_cluster/global_r49",
                               "/root/HPT/data/zarr_mujoco_metaworld_task3_resnet_traj100",
                               "mujoco_metaworld_task3", dev)
    batch = make_batch(d, 0, dev)
    local = compute_local_grad(pol, "mujoco_metaworld_task3", batch)
    uplink, trunk = split_uplink(local)
    p(f"[setup] trunk 梯度 {len(trunk)} 张量 / "
      f"{sum(v.numel() for v in trunk.values())/1e6:.3f}M")

    def run(tag, target, steps, lr, match_mode):
        t0 = time.time()
        r = invert_observation(pol, "mujoco_metaworld_task3", batch, trunk,
                               target=target, steps=steps, lr=lr, device=dev,
                               match_mode=match_mode, seed=7, verbose=True)
        curve = r["match_loss_curve"]
        if "corr_mean" in r:
            m = (f"corr={r['corr_mean']:.4f} (基线 {r['baseline_corr_mean']:.4f}) "
                 f"rel_l2={r['rel_l2']:.4f}")
        else:
            m = (f"cos={r['cos']:.4f} (基线 {r['baseline_cos']:.4f}) "
                 f"rel_l2={r['rel_l2']:.4f}")
        p(f"  [{tag}] {target}/{match_mode}/steps={steps}/lr={lr}: {m}  "
          f"match {curve[0]:.5f}→{curve[-1]:.5f} (min {min(curve):.5f})  "
          f"common={r['n_common_keys']}  {time.time()-t0:.0f}s")
        return r

    hr("A. 低维对照（动作 / 状态）")
    run("A1", "action", 300, 0.05, "dense")
    run("S1", "state", 300, 0.05, "dense")

    hr("B. 视觉：多配置排除'优化不足'")
    run("V2", "image", 300, 0.01, "dense")
    run("V3", "image", 300, 0.05, "all")
    run("V1", "image", 600, 0.05, "dense")

    hr("C. 结论")
    p("  若 V1/V2/V3 的 corr 都 ≪ 0.1，而 A1/S1 的 cos ≫ 0.9，")
    p("  ⇒ 视觉反演失败是**维度诅咒**（25088 维潜变量 vs 17.1M 维梯度），")
    p("     不是步数/学习率没调好 ⇒ 结论稳健。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        p(traceback.format_exc())
    finally:
        try:
            with open("/root/HPT/output/probe4.log", "w", encoding="utf-8") as f:
                f.write(OUT.getvalue())
        except Exception:
            pass
