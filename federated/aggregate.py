"""FedVLA 服务器端聚合（FedAvg / EDA）。

每个联邦轮结束后调用一次：把 4 个客户端的 `model.pth`（完整 state_dict）读出，
只平均 **trunk** 参数（键前缀 `trunk.`），然后写回全局目录：

  <out>/trunk.pth    —— trunk-only state_dict（键前缀 `trunk.`，对应 Policy.trunk）
  <out>/model.pth    —— 完整 state_dict（平均后的 trunk + 各客户端 stem/head 并集），用于最终 eval
  <out>/config.yaml  —— 从 client0 复制（HPT init_policy 用它实例化同架构）
  <out>/config.json  —— 若存在也复制

为什么只聚合 trunk：HPT 的 stem/head 是 **按域（domain）的 ModuleDict**（stems.<domain>_*、
heads.<domain>），不同客户端的域不同，无法跨客户端平均，且 FedVLA 本就是"trunk 共享聚合、
stem/head 个性化"。所以全局模型 = 平均 trunk + 各客户端 stem/head 并集。

两种聚合模式
------------
  * FedAvg（默认）：所有层、所有参数用同一组（可样本加权的）权重。
  * EDA（`--eda`）：**逐层**加权 —— 每个 MoE 层用自己那一行 EDA 权重，
    权重来自各客户端的专家选择计数（`expert_counts.pt`）。
    这就是 H2「按簇/中心性聚合更优」的实验路径。

依赖：torch（在 hpt 环境里跑）。用法见 federated/README.md。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import (aggregate_state_dicts, block_indices_from_names,  # noqa: E402
                 cluster_clients, cluster_report, eda_cluster_weights,
                 eda_diagnostics, eda_weights, format_weight_report)
from comm_channel import Quantizer  # noqa: E402


def _is_shared_param(key: str) -> bool:
    """这个键是否属于**上行共享**的参数（会经过链路传输）。

    FedVLA 的设计是「trunk 共享聚合、stem/head 个性化」：
      - `stems.<domain>_*` / `heads.<domain>` 是**个性化**参数，留在客户端本地不上传；
      - `trunk.*`、`normalizer.<domain>`、`modalities_tokens.*` 是共享参数，会走链路。

    所以量化（模拟链路压缩）只应作用在共享参数上 —— 这与 comm_cost.py 的
    「上行 = trunk 参数量」口径一致。若把 stems/heads 也量化了，
    等于把客户端本地的个性化参数也压了一遍，物理上不成立。
    """
    return not (key.startswith("stems.") or key.startswith("heads."))


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到客户端权重：{path}")
    # weights_only=False：HPT 的 model.pth 是普通 state_dict，torch>=2.1 默认即可载入
    return torch.load(path, map_location="cpu", weights_only=False)


def _is_trunk_key(k: str) -> bool:
    return k.startswith("trunk.")


def average_trunk(client_states: list[dict], weights: list[float] | None = None) -> dict:
    """对 trunk 参数做（可选样本加权的）均值。

    client_states[i] 是完整 state_dict；trunk 键形如 `trunk.trunk.xxx`。
    返回的字典键同 client_states 里的 trunk 键（带 `trunk.` 前缀），供直接写回完整 model.pth。
    """
    trunk_keys = [k for k in client_states[0] if _is_trunk_key(k)]
    if not trunk_keys:
        raise KeyError("未在客户端 state_dict 中找到任何 `trunk.` 前缀参数；请检查 HPT 版本。")

    if weights is None:
        weights = [1.0] * len(client_states)
    assert len(weights) == len(client_states), "weights 与客户端数不一致"
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights 之和必须 > 0")
    wnorm = [w / wsum for w in weights]

    avg: dict = {}
    for k in trunk_keys:
        # 形状必须一致才能平均
        try:
            stacked = torch.stack([s[k] for s in client_states], dim=0)
        except RuntimeError as e:
            raise RuntimeError(f"客户端 trunk 参数形状不一致，键={k}: {e}")
        avg[k] = sum(wnorm[i] * client_states[i][k] for i in range(len(client_states)))
    return avg


def average_trunk_eda(client_states: list[dict], weights: torch.Tensor,
                      block_map: list[int]) -> dict:
    """EDA 逐层加权聚合 trunk。

    weights  : (L, N)，第 r 行对应 block_map[r] 号 transformer block
    block_map: 行号 -> transformer block 号

    非 MoE block 的参数与共享参数用"沿层平均"的权重
    （等价于整体中心性权重；见 eda.py 的说明）。
    """
    trunk_keys = [k for k in client_states[0] if _is_trunk_key(k)]
    if not trunk_keys:
        raise KeyError("未在客户端 state_dict 中找到任何 `trunk.` 前缀参数；请检查 HPT 版本。")

    sub = [{k: v for k, v in sd.items() if _is_trunk_key(k)} for sd in client_states]
    avg = aggregate_state_dicts(sub, weights=weights, num_layers=weights.shape[0],
                                 block_map=block_map)
    # 键前缀在 aggregate_state_dicts 里被保留（我们只截取了 trunk.* 的子集）
    return avg


def load_expert_counts(client_dirs: list[str]):
    """读各客户端的 expert_counts.pt，返回 (V, names)。

    V: (N, L, K) 的选择计数张量；names: 对应的模块路径列表。
    """
    mats, names = [], None
    for d in client_dirs:
        p = os.path.join(d, "expert_counts.pt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"找不到专家计数文件：{p}\n"
                f"  EDA 模式需要客户端开启 FEDVLA_MOE=1 并导出 expert_counts.pt。")
        blob = torch.load(p, map_location="cpu", weights_only=False)
        M = blob["counts"] if isinstance(blob, dict) else blob
        if M.dim() != 2:
            raise ValueError(f"{p}: counts 应为 (L, K)，实际 {tuple(M.shape)}")
        mats.append(M.double())
        if names is None:
            names = list(blob["names"]) if isinstance(blob, dict) else []
    L = mats[0].shape[0]
    for i, m in enumerate(mats):
        if m.shape != mats[0].shape:
            raise ValueError(f"客户端 {i} 的专家计数形状 {tuple(m.shape)} "
                             f"与客户端 0 的 {tuple(mats[0].shape)} 不一致")
    V = torch.stack(mats, dim=0)          # (N, L, K)
    return V, (names or [])



def _tensor_equal_shape(vals: list) -> bool:
    shp = None
    for v in vals:
        if not torch.is_tensor(v):
            return False
        if shp is None:
            shp = tuple(v.shape)
        elif tuple(v.shape) != shp:
            return False
    return True


def build_global_model(client_states: list[dict], trunk_avg: dict,
                       weights: list[float] | None = None) -> dict:
    """用 client0 的完整 state_dict 打底，写入平均 trunk，并并集其余客户端的非 trunk 键。

    为什么不能只并集 `stems.` / `heads.`：
        门0 实测 HPT 完整 state_dict 的顶层前缀分布为
        `trunk(224) / stems(18) / normalizer(12) / heads(8) / modalities_tokens(2)`。
        其中 `normalizer.<domain>`、`modalities_tokens.<modality>` 同样是**按域/按模态分开存**的，
        异构客户端（task1/task2/task3 各一个）只并集 stems/heads 会导致：
          - 全局 model.pth 缺少其它域的 normalizer → 最终 eval 在该域上直接崩；
          - modalities_tokens 缺失 → trunk 里依赖 modality token 的分支失效。
        所以这里对**所有非 trunk 键**做并集。

    同名键的处理：
        形状一致 → 按权重平均（normalizer 统计量、共享 modality token 这类同形参数平均是合理的）；
        形状不一致 → 取第一个存在的（不平均，避免破坏结构），并打印警告。
    """
    full = dict(client_states[0])  # 浅拷贝，避免改到原 dict
    for k, v in trunk_avg.items():
        full[k] = v

    if weights is None:
        weights = [1.0] * len(client_states)
    wsum = sum(weights)
    if wsum <= 0:
        raise ValueError("weights 之和必须 > 0")
    wnorm = [w / wsum for w in weights]

    # 收集所有非 trunk 键
    groups: dict = {}
    for i, s in enumerate(client_states):
        for k, v in s.items():
            if _is_trunk_key(k):
                continue
            groups.setdefault(k, []).append((i, v))

    conflicts = []
    for k, items in groups.items():
        vals = [v for _, v in items]
        if len(items) == 1:
            full[k] = vals[0]
            continue
        if _tensor_equal_shape(vals):
            full[k] = sum(wnorm[i] * v for i, v in items)
        else:
            full[k] = vals[0]
            conflicts.append(k)

    if conflicts:
        print(f"[aggregate][WARN] {len(conflicts)} 个同名非 trunk 键形状不一致，已取首个客户端的值：")
        for k in conflicts[:10]:
            print(f"           - {k}")
    print(f"[aggregate] 非 trunk 键并集完成，共 {len(groups)} 个键"
          f"（前缀分布：{_prefix_hist(groups)}）")
    return full


def _prefix_hist(groups: dict) -> list:
    from collections import Counter
    return Counter(k.split(".")[0] for k in groups).most_common(10)


def extract_trunk_only(trunk_avg: dict) -> dict:
    """把完整 state_dict 里的 `trunk.trunk.xxx` 变成 trunk.pth 需要的 `trunk.xxx`。

    Policy.trunk 是 ModuleDict（内部键名 `trunk`），其 state_dict 键为 `trunk.xxx`；
    load_trunk 用 `self.trunk.load_state_dict(torch.load(path))` 载入。
    """
    out = {}
    for k, v in trunk_avg.items():
        assert k.startswith("trunk."), f"意外的 trunk 键：{k}"
        out[k[len("trunk."):]] = v  # 'trunk.trunk.xxx' -> 'trunk.xxx'
    return out


def extract_extras(full: dict) -> dict:
    """挑出**不属于 trunk、但属于策略顶层且跨客户端同形**的参数。

    为什么要单独导出：
        源码确认 `self.modalities_tokens` 是 `nn.ParameterDict`（按 modality 命名），
        `self.action_tokens` 条件式为 ParameterDict（按 domain 命名）——
        它们**不在 `self.trunk` 内部**，所以 trunk.pth 里没有它们。
        而 `load_trunk` 只载入 trunk → 客户端续训时模态 token 会保持随机初始化，
        导致同一个共享 trunk 在各客户端拿到不一致的输入。

        本函数导出这些键到 `extras.pth`，补丁后的 `load_trunk` 会顺带载入
        （用 policy.load_state_dict(strict=False)，键路径 `modalities_tokens.<modality>` 天然匹配）。
    """
    out = {}
    for k, v in full.items():
        if k.startswith("modalities_tokens.") or k.startswith("action_tokens."):
            out[k] = v
    return out



def aggregate(client_dirs: list[str], out_dir: str, weights: list[float] | None = None,
              eda_mode: bool = False, client_tasks: list[str] | None = None,
              eda_cluster: bool = False, num_clusters: int | None = None,
              drop_idx: list[int] | None = None,
              quant_scheme: str | None = None,
              prev_global: str | None = None,
              comm_report: str | None = None):
    """服务器端聚合。

    第4阶段的通信损伤注入（可选，三个参数全默认 None → 行为与原来完全一致）：
      * drop_idx     —— 这些客户端的上行更新在本轮**丢失**（信道中断/设备掉线）。
                        它们不参与本轮平均；同时其独占域的 stem/head / normalizer
                        必须从**上一轮全局模型**回退补齐（prev_global），
                        否则全局模型缺域 → 评估直接崩。
      * quant_scheme —— 上行共享参数按该量化方案压缩后再聚合（如 int8_g）。
      * prev_global  —— 丢包时用于回退补齐的上一轮全局模型目录（含 model.pth）。
      * comm_report  —— 通信量统计（每客户端字节数、总字节数）落盘路径。
    """
    os.makedirs(out_dir, exist_ok=True)
    client_models = [os.path.join(d, "model.pth") for d in client_dirs]
    states = [_load_state(m) for m in client_models]

    # ===== 第4阶段：上行链路损伤 =====
    comm_info = {"dropped": [], "kept": list(range(len(states))),
                  "quant_scheme": quant_scheme, "bytes_per_client": [],
                  "total_bytes": 0}

    # ===== 上行通信量：**无条件**统计（2026-11-18 补）=====
    # 为什么必须无条件：原来的字节统计写在 `if quant_scheme:` 里，
    # 于是 fp32 基线与三个丢包配置的 `total_bytes` 全是 null
    # ⇒ 论文的"上行流量–任务性能"曲线**缺基线点**（只有量化三档有横坐标）。
    # 这里按「所有客户端的共享参数都按当前方案上传」口径算 offered 流量，
    # 与丢包/量化无关，两个口径都写进 comm_report：
    #   offered_bytes_per_round —— 供给量（信道面对的负载）
    #   total_bytes             —— 实际送达量（丢包后只剩 kept 客户端）
    _bits = 32
    if quant_scheme:
        _bits = Quantizer(quant_scheme).spec["bits"]
    _byte_q = Quantizer(quant_scheme) if quant_scheme else Quantizer("fp32")
    _per_client_offered = [
        _byte_q.theoretical_bytes({k: v for k, v in st.items()
                                    if _is_shared_param(k)})
        for st in states]
    comm_info["offered_bytes_per_client"] = _per_client_offered
    comm_info["offered_bytes_per_round"] = int(sum(_per_client_offered))
    comm_info["uplink_bits"] = _bits
    print(f"[aggregate] 上行流量 offered = "
          f"{comm_info['offered_bytes_per_round']/1048576:.2f} MiB/轮"
          f"（{_bits} bit × {len(states)} 客户端，共享参数口径）")

    if drop_idx:
        drop_set = set(int(i) for i in drop_idx)
        keep = [i for i in range(len(states)) if i not in drop_set]
        if not keep:
            raise ValueError("所有客户端都被丢弃，无法聚合（至少要保留一路）")
        if not prev_global:
            raise ValueError(
                "丢包必须提供 prev_global：被丢弃客户端独占的 stem/head/normalizer "
                "需要从上一轮全局模型回退补齐，否则全局模型缺域 → 评估直接崩。")
        if eda_mode:
            # 丢包 + EDA 是**兼容**的，而且语义天然自洽：
            #   被丢弃客户端的 expert_counts.pt 本轮也没上传（它属于上行内容），
            #   所以簇划分与权重只用**本轮收到的**客户端重算 ——
            #   等价于"服务器这一轮只收到了 keep 这些客户端的一切"。
            #   代价：簇结构可能随丢包波动（簇数由 elbow 启发式在 keep 子集上重定），
            #   论文里需注明这一点。
            print(f"[aggregate][通信损伤] 丢包 + EDA：簇结构将在本轮收到的 "
                  f"{len(keep)} 个客户端上重算")
        comm_info["dropped"] = sorted(drop_set)
        comm_info["kept"] = keep
        states = [states[i] for i in keep]
        client_models = [client_models[i] for i in keep]
        client_dirs = [client_dirs[i] for i in keep]
        if weights is not None:
            weights = [weights[i] for i in keep]
        if client_tasks is not None:
            client_tasks = [client_tasks[i] for i in keep]
        print(f"[aggregate][通信损伤] 本轮丢弃客户端 {sorted(drop_set)}，"
              f"参与聚合 {keep}")

    if quant_scheme:
        q = Quantizer(quant_scheme)
        new_states = []
        bytes_list = []
        # ===== 量化自检（2026-11-18 新增，铁律 A：silent failure 必须显式自检）=====
        # 为什么必须有这层闸：第4阶段通信扫描里 int8_g / int4_g / bit1 三个配置
        # 实测成功率**全为 0.000**（3 seeds × 20 rollouts 全 0），
        # 而离线在真实模型上模拟的 int8_g 相对误差只有 **0.005**（几乎无损）。
        # 0.5% 的权重扰动不可能把策略从 0.75 打到 0.00 ⇒ 两者矛盾 ⇒ 强烈指向
        # "量化链路把权重搞坏了"（NaN / scale 错位 / 统计量错配）而不是信道现象。
        # 没有这层闸 ⇒ 烧 8 小时算力得到 3 个 0.000，且零报错。
        # 现在：NaN/Inf 或相对误差超阈值 → **立刻抛错中止**，不产出不可信的曲线。
        _LIMIT = {"int8_g": 0.10, "int4_g": 0.35, "int2_g": 1.00, "bit1": 1.50}
        limit = _LIMIT.get(quant_scheme, 1.00)
        qerrs = []
        orig_states = states          # 量化前的原始客户端状态（漂移诊断要用）
        for ci, st in enumerate(states):
            qsd = {}
            for k, v in st.items():
                # 只压共享参数（stems/heads 是个性化的，不上行）
                if torch.is_tensor(v) and _is_shared_param(k) and v.is_floating_point():
                    qsd[k] = q.apply_tensor(v)
                else:
                    qsd[k] = v
            # --- 自检：NaN/Inf + 实际相对误差 ---
            num = 0.0
            den = 0.0
            bad = []
            for k, v in st.items():
                if not (torch.is_tensor(v) and _is_shared_param(k) and v.is_floating_point()):
                    continue
                qv = qsd[k]
                if torch.isnan(qv).any() or torch.isinf(qv).any():
                    bad.append(k)
                num += float(torch.linalg.norm((qv - v).float()))
                den += float(torch.linalg.norm(v.float()))
            rel = (num / den) if den > 0 else 0.0
            qerrs.append(rel)
            print(f"[aggregate][通信损伤] 量化自检 client{ci}: "
                  f"相对误差={rel:.6f}（阈值 {limit}）"
                  + (f" NaN/Inf 键={bad[:3]}" if bad else ""))
            if bad:
                raise ValueError(
                    f"[量化自检] 方案 {quant_scheme} 在 client{ci} 上产生 NaN/Inf"
                    f"（例：{bad[:3]}）—— 拒绝聚合（否则会得到全 0 的假曲线）")
            if rel > limit:
                raise ValueError(
                    f"[量化自检] 方案 {quant_scheme} 在 client{ci} 上的相对误差 "
                    f"{rel:.6f} 超过阈值 {limit} —— 拒绝聚合"
                    f"（离线基准：int8_g≈0.005 / int4_g≈0.089 / bit1≈0.557）")
            new_states.append(qsd)
            bytes_list.append(q.theoretical_bytes(
                {k: v for k, v in st.items() if _is_shared_param(k)}))
        states = new_states
        # ---- 每轮「量化→全局模型」的漂移诊断（误差累积的直接证据）----
        # 为什么会累积：客户端每轮从**上一轮已被量化聚合过的**全局权重继续训练，
        # 所以量化误差不是每轮重置，而是逐轮叠加。
        # 若漂移随轮数线性增长到 O(1)，就能解释"成功率逐步塌到 0"。
        # 这里按等权近似（EDA 模式下权重逐层不同，仅作趋势诊断）。
        try:
            _tk = [k for k in orig_states[0] if _is_trunk_key(k)]
            _un = average_trunk(orig_states, None)
            _qt = average_trunk(new_states, None)
            _num = sum(float(torch.linalg.norm((_qt[k].float() - _un[k].float())))
                       for k in _tk)
            _den = sum(float(torch.linalg.norm(_un[k].float())) for k in _tk)
            drift = (_num / _den) if _den > 0 else 0.0
        except Exception as e:  # noqa: BLE001
            print(f"[aggregate][通信损伤][WARN] 漂移诊断失败：{e}")
            drift = float("nan")
        print(f"[aggregate][通信损伤] 量化→全局模型漂移 = {drift:.6f}（本轮，等权近似）")
        comm_info["quant_drift"] = drift
        try:
            _drift_file = os.path.join(out_dir, "quant_drift.jsonl")
            _rnd = 0
            if os.path.exists(_drift_file):
                with open(_drift_file, "r", encoding="utf-8") as fh:
                    _rnd = sum(1 for _ in fh)
            with open(_drift_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"round": _rnd + 1,
                                     "scheme": quant_scheme, "drift": drift,
                                     "rel_err_max": max(qerrs) if qerrs else 0.0},
                                    ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[aggregate][通信损伤][WARN] 漂移诊断落盘失败：{e}")
        comm_info["quant_rel_err_max"] = max(qerrs) if qerrs else 0.0
        comm_info["bytes_per_client"] = bytes_list
        comm_info["total_bytes"] = int(sum(bytes_list))
        print(f"[aggregate][通信损伤] 上行量化方案 = {quant_scheme}，"
              f"单客户端 {bytes_list[0]/1048576:.2f} MiB，"
              f"本轮上行合计 {comm_info['total_bytes']/1048576:.2f} MiB")

    if eda_mode:
        print(f"[aggregate] EDA 模式：客户端数={len(states)}"
              f"{'（簇感知：簇内加权 + 簇间等权）' if eda_cluster else '（中心性加权）'}")
        V, names = load_expert_counts(client_dirs)
        block_map = block_indices_from_names(names) if names else list(range(V.shape[1]))
        clusters = None
        if eda_cluster:
            clusters = cluster_clients(V, num_clusters=num_clusters)
            print(f"[aggregate] 客户端分簇结果: "
                  f"{dict(zip([f'c{i}' for i in range(len(states))], clusters))}")
            print(cluster_report(V, clusters, client_tasks))
            w = eda_cluster_weights(V, clusters)
        else:
            w = eda_weights(V)                    # (L, N)
        if len(block_map) != w.shape[0]:
            raise ValueError(f"从专家计数名解析出 {len(block_map)} 个 block，"
                             f"但权重的层数是 {w.shape[0]}；两者必须一致")
        print(f"[aggregate] EDA 权重形状={tuple(w.shape)}  对应 block={block_map}")
        print(format_weight_report(w, client_tasks or [f"c{i}" for i in range(len(states))],
                                   layer_names=[f"block{b}" for b in block_map]))
        diag = eda_diagnostics(V, client_tasks)
        # 非 trunk 参数（normalizer 统计量 / modalities_tokens）也用 EDA 的平均权重，
        # 保证"整份模型"用的是同一套客户端重要性判断。
        weights = diag["mean_weights"].tolist()
        trunk_avg = average_trunk_eda(states, w, block_map)
        # 把权重与诊断落盘，论文表格/图直接引用
        torch.save({"weights": w, "block_map": block_map,
                    "counts": V, "names": names, "clusters": clusters},
                   os.path.join(out_dir, "eda_weights.pt"))
        with open(os.path.join(out_dir, "eda_report.txt"), "w", encoding="utf-8") as f:
            f.write(format_weight_report(
                w, client_tasks or [f"c{i}" for i in range(len(states))],
                layer_names=[f"block{b}" for b in block_map]))
            f.write("\n\n平均权重: ")
            f.write(", ".join(f"c{i}={v:.4f}" for i, v in
                              enumerate(diag["mean_weights"].tolist())))
            f.write("\n")
            if clusters is not None:
                f.write("\n簇划分: " + str(clusters) + "\n")
                f.write(cluster_report(V, clusters, client_tasks) + "\n")
            if client_tasks:
                f.write("同任务相似度: " + ", ".join(f"{v:.4f}" for v in diag["within_task_sim"]))
                f.write("\n异任务相似度: " + ", ".join(f"{v:.4f}" for v in diag["across_task_sim"]))
                f.write("\n")
    else:
        print(f"[aggregate] FedAvg 模式：客户端数={len(states)}  权重={weights}")
        trunk_avg = average_trunk(states, weights)

    full = build_global_model(states, trunk_avg, weights)

    # ===== 丢包补偿：把被丢弃客户端独占的键从上一轮全局模型回退补齐 =====
    # 为什么必须补：某客户端被丢弃 → 它独占域的 stems./heads./normalizer.<domain>
    #   在本轮 states 里缺席 → 全局模型缺域 → 评估该域时 HPT 直接崩。
    # 为什么用上一轮全局模型是对的：真实 FL 服务器本来就持有上一轮的全局模型，
    #   丢包的客户端只是"这一轮没贡献新信息"，服务器沿用旧值继续服务，
    #   这与真实系统的行为一致（也正是"陈旧更新/staleness"的温和版本）。
    if drop_idx and prev_global:
        prev_path = os.path.join(prev_global, "model.pth")
        if os.path.exists(prev_path):
            prev_sd = _load_state(prev_path)
            restored = [k for k in prev_sd if k not in full]
            for k in restored:
                full[k] = prev_sd[k]
            print(f"[aggregate][通信损伤] 从上一轮全局模型回退补齐 {len(restored)} 个"
                  f"被丢弃客户端独占的键（例：{restored[:3]}）")
        else:
            print(f"[aggregate][通信损伤][WARN] 找不到上一轮全局模型 {prev_path}，"
                  f"被丢弃客户端独占的域将缺失 —— 评估该域会崩！")

    trunk_only = extract_trunk_only(trunk_avg)

    torch.save(full, os.path.join(out_dir, "model.pth"))
    torch.save(trunk_only, os.path.join(out_dir, "trunk.pth"))
    extras = extract_extras(full)
    if extras:
        torch.save(extras, os.path.join(out_dir, "extras.pth"))
        print(f"[aggregate] 已写出 extras.pth（{len(extras)} 个顶层共享参数）："
              f"{sorted(extras.keys())}")
    # 复制 config.yaml / config.json（HPT init_policy 与 run_eval 都需要）
    for fname in ("config.yaml", "config.json"):
        src = os.path.join(client_dirs[0], fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
            print(f"[aggregate] 复制 {fname}")
    print(f"[aggregate] 已写出全局模型 -> {out_dir}  (trunk.pth / model.pth / config*)")

    if comm_report:
        # 注意：不要再在这里 `import json` —— 函数内有 import 会让 json 变成
        # aggregate() 的**局部名**，导致本函数前面用 json.dumps 时报
        # "cannot access local variable 'json'"（2026-11-18 单测实测踩到）。
        with open(comm_report, "w", encoding="utf-8") as f:
            json.dump(comm_info, f, ensure_ascii=False, indent=2)
        print(f"[aggregate][通信损伤] 通信量统计 -> {comm_report}")


def main(argv=None):
    p = argparse.ArgumentParser(description="FedVLA 聚合（FedAvg / EDA）")
    p.add_argument("--clients", nargs="+", required=True,
                   help="各客户端输出目录（含 model.pth），空格分隔")
    p.add_argument("--out", required=True, help="全局模型输出目录")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="可选：各客户端样本量权重（FedAvg 样本加权）；省略则等权")
    p.add_argument("--eda", action="store_true",
                   help="用 EDA（专家驱动聚合）替代等权 FedAvg：逐层按专家计数相似度加权")
    p.add_argument("--eda-cluster", action="store_true",
                   help="簇感知聚合：先分簇，簇内中心性加权 + 簇间等权（需配合 --eda）")
    p.add_argument("--num-clusters", type=int, default=None,
                   help="簇感知聚合的目标簇数；省略则自动")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="可选：各客户端对应的任务名，用于 EDA 诊断报告")
    # ===== 第4阶段：通信损伤 =====
    # ⚠️ 2026-11-08 修正：原为 type=int，调用方传逗号串 "0,1" 会直接
    #    argparse 报错 "invalid int value: '0,1'" ⇒ 整轮聚合失败。
    #    c_drop25（只丢 1 个客户端）侥幸通过，c_drop50/c_drop75 静默失败。
    #    改成字符串 + 手工解析，同时兼容 "0 1" 与 "0,1" 两种写法。
    p.add_argument("--drop-clients", nargs="*", default=None,
                   help="本轮上行丢失的客户端索引（0 起）；不参与聚合。"
                        "支持空格分隔 '0 1' 或逗号分隔 '0,1'")
    p.add_argument("--quant-scheme", default=None,
                   help="上行共享参数的量化方案（fp32/fp16/int8_g/int4_g/int2_g/bit1/sparse1pct）")
    p.add_argument("--prev-global", default=None,
                   help="上一轮全局模型目录；丢包时用于回退补齐被丢弃客户端独占的键")
    p.add_argument("--comm-report", default=None,
                   help="通信量统计 JSON 的落盘路径")
    args = p.parse_args(argv)

    if args.weights is not None and len(args.weights) != len(args.clients):
        p.error("--weights 数量必须与 --clients 一致")
    if args.tasks is not None and len(args.tasks) != len(args.clients):
        p.error("--tasks 数量必须与 --clients 一致")
    if args.eda_cluster and not args.eda:
        p.error("--eda-cluster 需要配合 --eda 一起使用")
    if args.drop_clients and not args.prev_global:
        p.error("--drop-clients 必须配合 --prev-global（否则被丢弃客户端独占的域会缺失）")

    # 统一解析：容忍 "0 1" / "0,1" / "0, 1"
    drop_idx = []
    for tok in (args.drop_clients or []):
        drop_idx.extend(int(x) for x in str(tok).replace(",", " ").split())
    if drop_idx:
        print(f"[aggregate][通信损伤] 本轮丢弃客户端索引 = {sorted(set(drop_idx))}")

    aggregate(args.clients, args.out, args.weights, eda_mode=args.eda,
              client_tasks=args.tasks, eda_cluster=args.eda_cluster,
              num_clusters=args.num_clusters,
              drop_idx=drop_idx, quant_scheme=args.quant_scheme,
              prev_global=args.prev_global, comm_report=args.comm_report)



if __name__ == "__main__":
    main()
