"""任务级非 iid 划分（纯逻辑，无 torch 依赖，可本地自测）。

FedVLA 第 2 阶段：4 个客户端分 3 个 Meta-World 任务，构造"每客户端 1 个任务"的异构。
HPT 的 `domains` 是逗号分隔字符串（`cfg.domains.split(",")`），所以一个客户端的任务
列表直接拼成 `mujoco_metaworld_task1,mujoco_metaworld_task2` 即可。

注意：HPT 单次 `hpt.run` 只用 `domains` 列表里的 **第一个** 域来构建数据集与 stem/head
（见 hpt/run.py：`domain = domain_list[0]`）。因此每个客户端应只挂 **一个** 任务，
异构通过"不同客户端挂不同任务"实现，而不是一个客户端挂多个任务。

用法：
    python partition.py --num-clients 4 --tasks task1 task2 task3 --scheme task_level
    python partition.py --assignment task1 task2 task3 task1     # 直接指定（最透明）
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

# HPT 的 Meta-World domain 命名（reach=task1 已验证；其余请到远程确认索引）。
# 用法：python -c "from hpt.envs.metaworld... " 或在 HPT 配置里查 mujoco_metaworld_taskN。
MW_DOMAIN = "mujoco_metaworld_task"  # 拼上数字即为 domain 名


def task_domain(task_idx: int) -> str:
    """任务序号 -> HPT domain 字符串。task_idx 从 1 开始。"""
    return f"{MW_DOMAIN}{task_idx}"


def assign_task_level(num_clients: int, tasks: List[str]) -> List[List[str]]:
    """每个客户端分配一个不同任务，循环分配。

    例：4 客户端 + 3 任务 -> [t1, t2, t3, t1]（最后一个是重复/重叠客户端）。
    """
    if not tasks:
        raise ValueError("tasks 不能为空")
    return [[tasks[i % len(tasks)]] for i in range(num_clients)]


def assign_iid(num_clients: int, tasks: List[str]) -> List[List[str]]:
    """IID 健全性检查：每个客户端拿全部任务（实际 HPT 只用第 1 个，等价于全训同一任务）。"""
    return [list(tasks) for _ in range(num_clients)]


def assign_isolated(num_clients: int, tasks: List[str]) -> List[List[str]]:
    """含孤立客户端：1 个客户端独占一个别的客户端都没有的任务，其余共享剩余任务。

    用于后续 H2（按簇聚合）场景；第 2 阶段 FedAvg 基线一般先用 task_level。
    """
    if num_clients < 2 or len(tasks) < 2:
        raise ValueError("isolated 方案至少需要 2 客户端 + 2 任务")
    isolated_task = tasks[0]
    shared = tasks[1:]
    out: List[List[str]] = [[isolated_task]]
    for i in range(1, num_clients):
        out.append(list(shared))
    return out


def client_domains(client_idx: int, assignment: List[List[str]]) -> str:
    """返回某客户端要传给 HPT `domains=` 的逗号分隔字符串。"""
    if not (0 <= client_idx < len(assignment)):
        raise IndexError(f"client_idx={client_idx} 超出分配范围 {len(assignment)}")
    return ",".join(assignment[client_idx])


def assignment_to_bash(assignment: List[List[str]], varname: str = "CLIENT_DOMAINS") -> str:
    """把分配结果渲染成可直接粘进 fedavg_config.sh 的 bash 数组。"""
    items = " ".join(f'"{",".join(c)}"' for c in assignment)
    return f'{varname}=( {items} )'


def main(argv=None):
    p = argparse.ArgumentParser(description="FedVLA 任务级非 iid 划分")
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"],
                   help="任务索引名（task1/task2/... 或数字 1/2/3）")
    p.add_argument("--scheme", choices=["task_level", "iid", "isolated"], default="task_level")
    p.add_argument("--assignment", nargs="+", default=None,
                   help="直接指定每个客户端的任务（覆盖 --scheme）。如: task1 task2 task3 task1")
    args = p.parse_args(argv)

    # 归一化 tasks：支持 数字(1) / task1 / mujoco_metaworld_task1 三种写法，统一成 HPT 全名
    import re

    def norm(t: str) -> str:
        if t.isdigit():
            return task_domain(int(t))
        m = re.match(r"^task(\d+)$", t)
        if m:
            return task_domain(int(m.group(1)))
        if t.startswith("mujoco_metaworld_task"):
            return t
        # 已是合法 domain 或自定义名，原样保留
        return t
    tasks = [norm(t) for t in args.tasks]

    if args.assignment:
        assignment = [[norm(t)] for t in args.assignment]
        if len(assignment) != args.num_clients:
            print(f"[warn] --assignment 给了 {len(assignment)} 个，但 --num-clients={args.num_clients}；以 assignment 为准")
    else:
        if args.scheme == "task_level":
            assignment = assign_task_level(args.num_clients, tasks)
        elif args.scheme == "iid":
            assignment = assign_iid(args.num_clients, tasks)
        else:
            assignment = assign_isolated(args.num_clients, tasks)

    print("=== 客户端 -> 任务分配 ===")
    for i, c in enumerate(assignment):
        print(f"  client {i}: domains='{client_domains(i, assignment)}'  ({c})")
    print()
    print("=== 粘进 fedavg_config.sh ===")
    print(assignment_to_bash(assignment))
    # 同时输出 json，方便其他脚本消费
    print()
    print("=== json ===")
    print(json.dumps(assignment))
    return assignment


if __name__ == "__main__":
    main()
