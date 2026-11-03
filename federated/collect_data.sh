#!/usr/bin/env bash
# FedVLA：采集异构客户端所需的 Meta-World 任务数据（现场渲染 + resnet 特征预计算）
# 用法：bash collect_data.sh
#
# 关键事实（已逐字核实 HPT 源码 hpt/dataset/local_traj_dataset.py，2026-11-01）：
#   - 缓存路径 = "data/zarr_" + dataset_name + dataset_encoder_postfix + dataset_postfix
#     = data/zarr_{domain}_resnet_traj100
#   - 判据（源码 L197）：load_from_cache = os.path.exists(dataset_path + "/data/action")
#     ⚠️ data/action 是【目录】（zarr 的 chunk 目录），不是文件！
#        用 `[ -f .../data/action ]` 判定会永远为假 —— 之前误报"采集失败"就是栽在这。
#   - ⚠️ 目录存在但缺 data/action 时（use_disk=False）：
#       走 ReplayBuffer.create_empty_numpy() → 数据只在内存，【不落盘】！
#       下次训练还要重新渲染 16 分钟。
#       所以必须先把这种空壳目录移走，让 HPT 走 create_empty_zarr 分支落盘。
#   - generate_dataset_rollouts 只认 env_names，不认 domains；
#     domains 只是缓存目录标签，env_names 决定真实任务。
set -u
cd /root/HPT || exit 1
export MUJOCO_GL=egl WANDB_MODE=offline HF_ENDPOINT=https://hf-mirror.com OMP_NUM_THREADS=8
PY=/root/miniconda3/envs/hpt/bin/python
EPISODE_CNT=100                                    # 每任务示范条数（论文协议：100）

# ---- 防重入锁 ----
# relay 队列重派 / 手动重复执行时，若已有采集进程在跑，直接退出，
# 否则两个进程会同时写同一个 zarr 目录，把数据写坏。
exec 9>/root/.collect_data.lock
if ! flock -n 9; then
  echo "[SKIP] 已有采集进程在运行（锁被占用），本次跳过"
  exit 0
fi

# 缓存是否真的可用：zarr 数组 data/action 第一维 > 0 且 episode_ends 非空。
# 只看文件/目录存在是不够的 —— 空壳目录会让 HPT 走内存分支，数据不落盘。
cache_ok () {
  local cache="$1"
  [ -d "$cache/data/action" ] || return 1
  "$PY" - "$cache" <<'PY' 2>/dev/null
import sys, zarr
try:
    g = zarr.open(sys.argv[1], mode="r")
    n = int(g["data/action"].shape[0])
    e = int(g["meta/episode_ends"].shape[0]) if "meta/episode_ends" in g else 0
except Exception:
    sys.exit(1)
sys.exit(0 if (n > 0 and e > 0) else 1)
PY
}

collect () {
  local domain="$1" env="$2"
  local cache="/root/HPT/data/zarr_${domain}_resnet_traj100"
  local log="/root/HPT/log_collect_${domain}.txt"

  if [ "${FORCE:-0}" = "1" ]; then
    if [ -e "$cache" ]; then
      local stale="${cache}.stale_$(date +%s)"
      echo "[FORCE] $cache 已存在，移到 $stale 后重新采集"
      mv "$cache" "$stale"
    fi
  elif cache_ok "$cache"; then
    echo "[SKIP] $domain 缓存已就绪：$cache ($(du -sh "$cache" | cut -f1))"
    return 0
  fi

  # 空壳/残缺缓存：移走而不是删（HPT 自己在落盘分支也会 rm -rf），
  # 移走的好处是失败后还能回滚，不丢任何东西。
  if [ -e "$cache" ]; then
    local stale="${cache}.stale_$(date +%s)"
    echo "[FIX] $cache 存在但不完整，移到 $stale（让 HPT 重新生成并落盘）"
    mv "$cache" "$stale"
  fi

  echo "=========== 采集 $domain <- $env ==========="
  local t0=$(date +%s)
  "$PY" -m hpt.run \
    env=mujoco_metaworld \
    domains="$domain" \
    +datasetsize=metaworld_1 \
    "dataset_generator_func.env_names=[\"$env\"]" \
    dataset.episode_cnt=$EPISODE_CNT \
    train.total_iters=1 train.total_epochs=1 \
    train.epoch_iters=1 train.validation_iters=1 \
    train.pretrained_dir=hf://liruiw/hpt-base \
    output_dir="/root/HPT/output/_collect_${domain}" \
    > "$log" 2>&1
  local rc=$?
  echo "[---] 用时 $(( $(date +%s) - t0 )) 秒  rc=$rc"

  if cache_ok "$cache"; then
    local n_eps
    n_eps=$("$PY" - "$cache" <<'PY2' 2>/dev/null
import sys, zarr
g = zarr.open(sys.argv[1], mode="r")
print(int(g["meta/episode_ends"].shape[0]))
PY2
)
    echo "[OK] $domain 数据就绪  $(du -sh "$cache" | cut -f1)  episode 数=$n_eps"
    if [ "$n_eps" -lt "$EPISODE_CNT" ]; then
      echo "[!!] $domain episode 数不足：$n_eps < $EPISODE_CNT（采集被中断？用 FORCE=1 重采）"
      return 1
    fi
  else
    echo "[!!] $domain 采集失败（缓存仍不可用），日志尾部："
    tail -n 30 "$log"
    return 1
  fi
}

# 只采集指定任务（逗号分隔，如 ONLY_TASKS=task1,task2）；默认全采。
# FORCE=1 时即使缓存已就绪也重新采集（用于修复"上次采集被中断、episode 数不足"的情况）。
ONLY_TASKS=${ONLY_TASKS:-"task1,task2,task3"}
ONLY_TASKS=${ONLY_TASKS// /,}
want () {
  case ",$ONLY_TASKS," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

want task1 && collect mujoco_metaworld_task1 reach-v2
want task2 && collect mujoco_metaworld_task2 push-v2
want task3 && collect mujoco_metaworld_task3 button-press-topdown-v2
echo "ALL_COLLECT_DONE"
