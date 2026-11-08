#!/usr/bin/env bash
# 第4阶段通信损伤注入的**云端真实权重干跑**。
#
# 目的：在烧 7 个配置 × 25 轮之前，用第3阶段的真实权重验证
#   1) aggregate.py 的 --drop-clients / --prev-global / --quant-scheme 在真实数据上不崩；
#   2) 被丢弃客户端（独占 task3）的域被上一轮全局模型正确回退补齐；
#   3) MoE 专家键一个不少（264 个，blocks 8~15）；
#   4) 通信量统计写出来了。
#
# 用 P3-C（异构+MoE+簇感知）的 client_{1,2,3}_r49（c0 已被 KEEP_ROUNDS 清理）
# + global_r48 作 prev_global。输出写到 /tmp，不碰原目录。
set -u
HPT=/root/miniconda3/envs/hpt/bin/python
FED=/root/HPT/federated
SRC=/root/HPT/output/p3_moe_eda_cluster
TMP=/tmp/comm_dryrun
rm -rf "$TMP"; mkdir -p "$TMP"

CLIENTS="$SRC/client_1_r49 $SRC/client_2_r49 $SRC/client_3_r49"

echo "==================== 1. 丢包：丢弃 client_2（独占 task3）===================="
$HPT "$FED/aggregate.py" \
  --clients $CLIENTS \
  --out "$TMP/out_drop" \
  --eda --eda-cluster \
  --tasks mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1 \
  --drop-clients 1 \
  --prev-global "$SRC/global_r48" \
  --comm-report "$TMP/comm_drop.json" 2>&1 | tail -n 25

echo
echo "==================== 2. 量化：int8_g ===================="
$HPT "$FED/aggregate.py" \
  --clients $CLIENTS \
  --out "$TMP/out_int8" \
  --eda --eda-cluster \
  --tasks mujoco_metaworld_task2 mujoco_metaworld_task3 mujoco_metaworld_task1 \
  --quant-scheme int8_g \
  --comm-report "$TMP/comm_int8.json" 2>&1 | tail -n 12

echo
echo "==================== 3. 结果体检 ===================="
$HPT - <<'PY'
import json, os, torch

TMP = "/tmp/comm_dryrun"
SRC = "/root/HPT/output/p3_moe_eda_cluster"

def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)

def moe_keys(sd):
    return [k for k in sd if "moe" in k.lower()]

def report(tag, path, expect_task3=True):
    print(f"--- {tag} ---")
    if not os.path.exists(path):
        print("   [FAIL] 输出不存在")
        return
    sd = load(path)
    mk = moe_keys(sd)
    blocks = sorted({int(k.split(".")[3]) for k in mk
                     if k.startswith("trunk.trunk.blocks.") and len(k.split(".")) > 4})
    print(f"   总键数 = {len(sd)}   MoE 键数 = {len(mk)}   MoE blocks = {blocks}")
    doms = sorted({k.split(".")[1] for k in sd if k.startswith("normalizer.")})
    print(f"   normalizer 覆盖的域 = {doms}")
    ok = True
    if len(mk) != 264:
        print("   [FAIL] MoE 键数不是 264")
        ok = False
    else:
        print("   [OK] MoE 键数 264，blocks 8~15")
    if blocks != list(range(8, 16)):
        print(f"   [FAIL] MoE 层不对：{blocks}")
        ok = False
    # normalizer 的实际键名形如
    #   normalizer.<domain>.params_dict.action.input_stats.mean
    # 所以判据按**前缀**查，而不是固定后缀
    if expect_task3 and not any(k.startswith("normalizer.mujoco_metaworld_task3.")
                                 for k in sd):
        print("   [FAIL] 缺 task3 的 normalizer（回退补齐失败）")
        ok = False
    else:
        print("   [OK] task3 的 normalizer 在（回退补齐成功）")
    if expect_task3 and not any(k.startswith("stems.mujoco_metaworld_task3")
                                 for k in sd):
        print("   [FAIL] 缺 task3 的 stems（回退补齐失败）")
        ok = False
    else:
        print("   [OK] task3 的 stems 在（回退补齐成功）")
    return ok

ok = report("丢包聚合", f"{TMP}/out_drop/model.pth")
ok = report("int8 量化聚合", f"{TMP}/out_int8/model.pth") and ok

print()
print("--- 通信量统计 ---")
for f in ("comm_drop.json", "comm_int8.json"):
    p = os.path.join(TMP, f)
    if os.path.exists(p):
        print(f"   {f}: {json.dumps(json.load(open(p)), ensure_ascii=False)[:220]}")
    else:
        print(f"   [FAIL] {f} 不存在")
        ok = False

print()
print("--- 量化误差核对（int8 应在 0.01 量级）---")
try:
    from importlib.machinery import SourceFileLoader
    cc = SourceFileLoader("cc", f"{'/root/HPT/federated'}/comm_channel.py").load_module()
    sd = load(f"{SRC}/global_r49/model.pth")
    for s in ("fp32", "fp16", "int8_g", "int4_g", "bit1"):
        q = cc.Quantizer(s)
        print(f"   {s:<8} rel_err = {q.mean_tensor_rel_err(sd):.4f}   "
              f"上行 = {q.theoretical_bytes({k: v for k, v in sd.items() if not k.startswith(('stems.','heads.'))})/1048576:.2f} MiB")
except Exception as e:
    print(f"   [WARN] 无法计算量化误差: {e}")

print()
print("DRYRUN_ALL_PASS" if ok else "DRYRUN_SOME_FAILED")
PY
