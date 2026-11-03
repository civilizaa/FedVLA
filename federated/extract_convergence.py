#!/usr/bin/env python3
"""FedVLA：从联邦训练日志提取收敛曲线（论文第 5 章）。

数据来源：run_fedavg.sh 写的 output/<ROOT>/log_client_<c>_r<r>.txt
每行形如： "Steps: 60. Train loss: 1.2345. Test loss: 1.1111"
（hpt/run.py 的 print，来自 train_test.train 的返回值）

输出：
  - 每个客户端每轮的 loss（画收敛曲线）
  - 全局聚合后的平均 loss（4 客户端均值 = 该轮的"联邦 loss"）
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# 形如： Steps: 60. Train loss: 1.2345. Test loss: 1.1111
STEP_RE = re.compile(r"Steps:\s*(\d+)\.\s*Train loss:\s*([0-9.eE+-]+)\.\s*Test loss:\s*([0-9.eE+-]+)")
# 有些版本只打 Train loss
STEP_RE2 = re.compile(r"Steps:\s*(\d+)\.\s*Train loss:\s*([0-9.eE+-]+)")


def parse_client_log(path):
    """返回 [(epoch, steps, train_loss, test_loss), ...]"""
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                    continue
                m2 = STEP_RE2.search(line)
                if m2:
                    out.append((int(m2.group(1)), float(m2.group(2)), None))
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="提取联邦收敛曲线")
    ap.add_argument("--root", default="/root/HPT/output/fedavg_hetero")
    ap.add_argument("--out", default="/root/HPT/output/convergence.json")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[!!] 找不到目录 {args.root}")
        return 2

    # 找所有 client 日志，按 (client, round) 归类
    per_client = defaultdict(dict)   # client -> {round: [(steps, loss, test)]}
    fn = re.compile(r"log_client_(\d+)_r(\d+)\.txt$")
    for name in os.listdir(args.root):
        m = fn.match(name)
        if not m:
            continue
        c, r = int(m.group(1)), int(m.group(2))
        recs = parse_client_log(os.path.join(args.root, name))
        if recs:
            per_client[c][r] = recs

    if not per_client:
        print(f"[!!] {args.root} 下没有可解析的 client 日志")
        return 2

    clients = sorted(per_client)
    rounds = sorted(set(r for c in clients for r in per_client[c]))
    print(f"解析到 {len(clients)} 个客户端，{len(rounds)} 轮")
    print()

    # 每轮的联邦平均 loss
    curve = []
    for r in rounds:
        losses, tests = [], []
        for c in clients:
            recs = per_client[c].get(r)
            if not recs:
                continue
            # 取该客户端该轮的最后一个 epoch 的 loss
            last = recs[-1]
            losses.append(last[1])
            if last[2] is not None:
                tests.append(last[2])
        if losses:
            curve.append({
                "round": r,
                "mean_train_loss": sum(losses) / len(losses),
                "mean_test_loss": (sum(tests) / len(tests)) if tests else None,
                "per_client_train_loss": {str(c): per_client[c][r][-1][1]
                                          for c in clients if r in per_client[c]},
            })

    print("轮次 | 联邦平均训练 loss | 各客户端")
    print("-" * 78)
    for item in curve:
        clients_str = " ".join(f"c{k}={v:.3f}" for k, v in item["per_client_train_loss"].items())
        print(f"{item['round']:>4} | {item['mean_train_loss']:>17.4f} | {clients_str}")

    result = {
        "root": args.root,
        "clients": clients,
        "rounds": rounds,
        "curve": curve,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print()
    print(f"已保存: {args.out}")
    print("CONVERGENCE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
