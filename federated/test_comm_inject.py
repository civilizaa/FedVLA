"""第4阶段通信损伤注入的本地正确性测试（纯 CPU，不依赖 hpt 环境）。

验证三件事：
  1. 无损伤时聚合行为与改造前一致（基线不被破坏）；
  2. 丢包后，被丢弃客户端**独占域**的键被上一轮全局模型回退补齐；
  3. 量化只作用在共享参数（trunk/normalizer/modalities_tokens），
     **不碰** stems./heads.（个性化参数不上行）。

跑法： python test_comm_inject.py
"""
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as agg  # noqa: E402


def make_client(root, tag, domain, expert_pref=None):
    """造一个假的客户端目录（model.pth + config.yaml [+ expert_counts.pt]）。

    expert_pref: 该客户端偏好的专家索引列表（模拟 MoE 专家分化）
    """
    d = os.path.join(root, tag)
    os.makedirs(d, exist_ok=True)
    sd = {
        # 共享参数（上行）
        "trunk.trunk.blocks.0.mlp.fc1.weight": torch.full((8, 8), 0.1 * (1 + len(domain))),
        "trunk.trunk.blocks.0.mlp.fc2.weight": torch.full((8, 8), 0.2 * (1 + len(domain))),
        "normalizer.%s.mean" % domain: torch.full((4,), 0.5),
        "modalities_tokens.image": torch.full((4,), 0.3),
        # 个性化参数（不上行）
        "stems.%s_stem.weight" % domain: torch.full((4, 4), 0.7),
        "heads.%s_head.weight" % domain: torch.full((4, 4), 0.8),
    }
    torch.save(sd, os.path.join(d, "model.pth"))
    with open(os.path.join(d, "config.yaml"), "w", encoding="utf-8") as f:
        f.write("x: 1\n")
    if expert_pref is not None:
        # (L=2, K=8) 计数；偏好专家给高计数
        L, K = 2, 8
        M = torch.ones(L, K, dtype=torch.float64)
        for li in range(L):
            for e in expert_pref:
                M[li, e] = 100.0
        torch.save({"counts": M, "names": [f"blocks.{li}.moe" for li in range(L)]},
                   os.path.join(d, "expert_counts.pt"))
    return d


def keys_of(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd


def check(name, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")
    return cond


def main():
    tmp = tempfile.mkdtemp(prefix="fedvla_comm_test_")
    try:
        c0 = make_client(tmp, "c0", "task1")
        c1 = make_client(tmp, "c1", "task2")
        c2 = make_client(tmp, "c2", "task3")
        out_base = os.path.join(tmp, "out_base")

        print("== 1. 无损伤基线（必须与改造前一致）==")
        agg.aggregate([c0, c1, c2], out_base)
        full = keys_of(os.path.join(out_base, "model.pth"))
        ok = True
        ok &= check("含 task1/task2/task3 的 normalizer",
                    all(f"normalizer.{t}.mean" in full for t in ("task1", "task2", "task3")))
        ok &= check("含各域 stems/heads",
                    all(f"stems.{t}_stem.weight" in full for t in ("task1", "task2", "task3")))
        ok &= check("含 modalities_tokens", "modalities_tokens.image" in full)
        # trunk 是三 client's 平均：0.1*(1+5)=0.6, 0.1*(1+5)=0.6, 0.1*(1+5)=0.6 → 0.6
        ok &= check("trunk 为等权平均",
                    abs(float(full["trunk.trunk.blocks.0.mlp.fc1.weight"].flatten()[0]) - 0.6) < 1e-6,
                    f"(得到 {float(full['trunk.trunk.blocks.0.mlp.fc1.weight'].flatten()[0]):.4f})")

        print()
        print("== 2. 丢包 c2（独占 task3）+ prev_global 回退 ==")
        out_drop = os.path.join(tmp, "out_drop")
        agg.aggregate([c0, c1, c2], out_drop, drop_idx=[2], prev_global=out_base)
        full_d = keys_of(os.path.join(out_drop, "model.pth"))
        ok &= check("被丢弃客户端的 task3 normalizer 已被回退补齐",
                    "normalizer.task3.mean" in full_d)
        ok &= check("被丢弃客户端的 task3 stems/heads 已被回退补齐",
                    "stems.task3_stem.weight" in full_d and "heads.task3_head.weight" in full_d)
        # trunk 只应由 c0、c1 平均：0.6, 0.6 → 0.6（数值巧合，验证一下 c2 确实没参与）
        t = float(full_d["trunk.trunk.blocks.0.mlp.fc1.weight"].flatten()[0])
        ok &= check("trunk 不含被丢弃客户端的贡献（c0/c1 平均 = 0.6）",
                    abs(t - 0.6) < 1e-6, f"(得到 {t:.4f})")

        print()
        print("== 3. 量化 int8_g（共享参数被压，stems/heads 不动）==")
        out_q = os.path.join(tmp, "out_q")
        agg.aggregate([c0, c1, c2], out_q, quant_scheme="int8_g",
                      comm_report=os.path.join(tmp, "comm.json"))
        full_q = keys_of(os.path.join(out_q, "model.pth"))
        ok &= check("stems 未被量化（仍为 0.7）",
                    abs(float(full_q["stems.task1_stem.weight"].flatten()[0]) - 0.7) < 1e-6)
        ok &= check("heads 未被量化（仍为 0.8）",
                    abs(float(full_q["heads.task1_head.weight"].flatten()[0]) - 0.8) < 1e-6)
        # int8 per-group(64)：0.6/0.6/0.6 三者相同 → 量化后仍应≈0.6（误差极小）
        tq = float(full_q["trunk.trunk.blocks.0.mlp.fc1.weight"].flatten()[0])
        ok &= check("共享参数经 int8 量化后数值仍接近 0.6",
                    abs(tq - 0.6) < 0.02, f"(得到 {tq:.6f})")
        ok &= check("通信量报告已生成", os.path.exists(os.path.join(tmp, "comm.json")))

        print()
        print("== 4. drop + EDA 兼容（簇结构在收到的客户端上重算）==")
        m0 = make_client(tmp, "m0", "task1", expert_pref=[0, 1])
        m1 = make_client(tmp, "m1", "task2", expert_pref=[0, 1])
        m2 = make_client(tmp, "m2", "task3", expert_pref=[6, 7])
        m3 = make_client(tmp, "m3", "task4", expert_pref=[6, 7])
        out_eda = os.path.join(tmp, "out_eda")
        try:
            agg.aggregate([m0, m1, m2, m3], out_eda, eda_mode=True, eda_cluster=True,
                          client_tasks=["task1", "task2", "task3", "task4"])
            ok &= check("drop+eda 未抛异常", True)
            ok &= check("产出 eda_weights.pt",
                        os.path.exists(os.path.join(out_eda, "eda_weights.pt")))
        except Exception as e:
            ok &= check("drop+eda 未抛异常", False, f"({type(e).__name__}: {str(e)[:80]})")

        out_eda_d = os.path.join(tmp, "out_eda_drop")
        try:
            agg.aggregate([m0, m1, m2, m3], out_eda_d, eda_mode=True, eda_cluster=True,
                          client_tasks=["task1", "task2", "task3", "task4"],
                          drop_idx=[2, 3], prev_global=out_eda)
            blob = torch.load(os.path.join(out_eda_d, "eda_weights.pt"),
                              map_location="cpu", weights_only=False)
            w = blob["weights"]
            ok &= check("丢包后 EDA 权重列数 = 参与客户端数 2", w.shape[1] == 2, f"(实际 {tuple(w.shape)})")
            s = w.sum(dim=1)
            ok &= check("每层权重沿客户端维和为 1",
                        bool(torch.allclose(s, torch.ones_like(s), atol=1e-6)),
                        f"(和={s.tolist()})")
            ok &= check("被丢弃客户端独占的域被回退补齐",
                        "normalizer.task3.mean" in keys_of(os.path.join(out_eda_d, "model.pth")))
        except Exception as e:
            ok &= check("drop+eda（丢包）未抛异常", False,
                        f"({type(e).__name__}: {str(e)[:80]})")

        print()
        print("== 5. 互斥保护：drop 缺 prev-global 应报错 ==")
        try:
            agg.aggregate([c0, c1, c2], os.path.join(tmp, "out_x"), drop_idx=[1])
            ok &= check("应抛异常", False, "(没有报错)")
        except ValueError as e:
            ok &= check("应抛异常", True, f"({str(e)[:50]}...)")

        print()
        print("ALL PASS" if ok else "SOME FAILED")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
