import os
import sys

import zarr

BASE = "/root/HPT/data"

for t in ["task1", "task2", "task3"]:
    p = f"{BASE}/zarr_mujoco_metaworld_{t}_resnet_traj100"
    try:
        g = zarr.open(p + "/data/language", mode="r")
        s = g[:]
        vals = set()
        for row in s:
            for v in row:
                vals.add(str(v))
        print(t, "->", sorted(vals)[:6], "共", len(vals), "种")
    except Exception as e:
        print(t, "ERR", type(e).__name__, e)

    # 同时统计 episode 数与 state 维度
    try:
        import numpy as np
        ends = np.load(p + "/meta/episode_ends.npy", allow_pickle=True)
        print("   episode_ends:", len(ends))
    except Exception as e:
        print("   episode_ends ERR", e)
    try:
        st = zarr.open(p + "/data/state", mode="r")
        print("   state shape:", st.shape)
    except Exception as e:
        print("   state ERR", e)
