#!/usr/bin/env python3
"""FedVLA：精确计算每个 domain 的 epoch_size（= len(DataLoader)），用于校准 LOCAL_TOTAL_ITERS。

为什么需要它
------------
HPT 的 `train.total_iters` 单位是【总 batch step 数】(源码 hpt/run.py)：
    train_steps = (epoch + 1) * len(train_loader)
    if train_steps > cfg.train.total_iters: break
所以"跑满 LOCAL_EPOCHS 个 epoch"需要 total_iters = LOCAL_EPOCHS × epoch_size - 1。
不同 Meta-World 任务的 episode 长度不同 → epoch_size 不同 → 必须逐任务测量。

本脚本不训练，只复刻 LocalTrajDataset + SequenceSampler 的长度计算：
    hpt/dataset/local_traj_dataset.py
        self.horizon = action_horizon + observation_horizon - 1 = 8
        val_mask = get_val_mask(n_episodes, val_ratio=0.1, seed=233)
        train_mask = ~val_mask; 前 episode_cnt(=100) 个之外置 False
        sampler = SequenceSampler(..., sequence_length=horizon, pad_before=0, pad_after=0)
        __len__ = len(sampler)
    hpt/utils/sampler.py
        create_indices(episode_ends, sequence_length, episode_mask, pad_before, pad_after)

    epoch_size = ceil(len(dataset) / batch_size=256)   # drop_last=False

自检：mujoco_metaworld_task1 的计算值必须等于集中式门1 实测的 15，
      对不上说明本脚本的复刻逻辑错了（而不是数据坏了）。
"""
import math
import os
import sys

import numpy as np
import zarr

sys.path.insert(0, "/root/HPT")
from hpt.utils.sampler import create_indices, get_val_mask  # noqa: E402

BASE = "/root/HPT/data"
DOMAINS = ["mujoco_metaworld_task1", "mujoco_metaworld_task2", "mujoco_metaworld_task3"]

HORIZON = 8          # action_horizon(8) + observation_horizon(1) - 1
VAL_RATIO = 0.1
SEED = 233           # LocalTrajDataset.__init__ 默认 seed
EPISODE_CNT = 100    # dataset.episode_cnt
BATCH = 256          # dataloader.batch_size
EXPECT_TASK1_EPOCH_SIZE = 15  # 集中式门1 实测值，用于自检


def probe(domain):
    cache = os.path.join(BASE, f"zarr_{domain}_resnet_traj100")
    g = zarr.open(cache, mode="r")
    ends = np.asarray(g["meta/episode_ends"][:])
    n = len(ends)

    val_mask = get_val_mask(n_episodes=n, val_ratio=VAL_RATIO, seed=SEED)
    train_mask = ~val_mask
    n_ep = int(min(EPISODE_CNT, n))
    val_mask[n_ep:] = False
    train_mask[n_ep:] = False
    if train_mask.sum() == 0:
        train_mask = val_mask

    idx = create_indices(
        ends,
        sequence_length=HORIZON,
        episode_mask=train_mask,
        pad_before=0,
        pad_after=0,
    )
    len_ds = len(idx)
    epoch_size = math.ceil(len_ds / BATCH)

    lens = np.diff(np.concatenate([[0], ends[:n_ep]]))
    print(
        f"{domain}: episodes_in_cache={n} used={n_ep} train_eps={int(train_mask.sum())} "
        f"len_dataset={len_ds} epoch_size={epoch_size} "
        f"ep_len[min/mean/max]={lens.min()}/{lens.mean():.1f}/{lens.max()}"
    )
    return epoch_size


if __name__ == "__main__":
    results = {}
    for d in DOMAINS:
        try:
            results[d] = probe(d)
        except Exception as e:  # noqa: BLE001
            print(f"{d}: ERR {type(e).__name__} {e}")
            results[d] = None

    print("---- summary ----")
    for d, v in results.items():
        print(f"{d}\tepoch_size={v}\tlocal_total_iters(3ep)={None if v is None else 3 * v - 1}")

    t1 = results.get("mujoco_metaworld_task1")
    if t1 is None:
        print("[!!] task1 探针失败 —— 无法自检，请人工核对上面的输出")
        sys.exit(2)
    if t1 != EXPECT_TASK1_EPOCH_SIZE:
        print(f"[!!] 自检失败：task1 计算 epoch_size={t1}，但集中式门1 实测为 "
              f"{EXPECT_TASK1_EPOCH_SIZE} —— 复刻逻辑有误，不要据此改配置")
        sys.exit(3)
    print(f"[OK] 自检通过：task1 epoch_size={t1} 与集中式门1 实测一致")
