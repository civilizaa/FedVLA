#!/usr/bin/env bash
# FedVLA 云端前置检查：一键确保代码到位 + 门0 + 门1
#
# 用法（AutoDL，任意目录）：
#     bash cloud_gates.sh
# 脚本会自动把 federated/ 放到 /root/HPT/federated/（run_fedavg.sh 期望的路径）。
#
# 门0 / 门1 都是纯 CPU 操作，**可以在无卡模式跑**（省 ¥0.9/小时）。
# 门2（异构续训烟雾）必须开卡，见 run_fedavg.sh。
#
# 设计要点：
#   - 不依赖 git（国内 git 镜像经常 504），raw 直取 + 多镜像降级
#   - 每次下载都校验文件大小与内容，杜绝 404 假文件（curl 对 404 返回 0）
set -u

HPT_PY=/root/miniconda3/envs/hpt/bin/python
HPT_DIR=/root/HPT
FEDVLA_DIR=/root/FedVLA
DEST="$HPT_DIR/federated"
REPO="civilizaa/FedVLA"
BRANCH="main"

# 镜像列表：按顺序尝试，某个成功就固定用它
MIRRORS=(
  "https://raw.githubusercontent.com/$REPO/$BRANCH"
  "https://raw.gitmirror.com/$REPO/$BRANCH"
  "https://ghproxy.net/https://raw.githubusercontent.com/$REPO/$BRANCH"
  "https://gh-proxy.com/https://raw.githubusercontent.com/$REPO/$BRANCH"
)
ACTIVE_MIRROR=""

# 若设置了 GITHUB_PAT，则优先走 API（私有仓库/限流时更可靠）
GITHUB_PAT="${GITHUB_PAT:-}"
API_BASE="https://api.github.com/repos/$REPO/contents/federated"

export MUJOCO_GL=egl
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export PYTHONPATH="$HPT_DIR:${PYTHONPATH:-}"

cd "$HPT_DIR" || { echo "找不到 $HPT_DIR"; exit 1; }

FILES=(aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md cloud_gates.sh)

# 判断文件是否有效：404 页面只有 14 字节，内容为 "404: Not Found"
is_valid() {
  local f="$1"
  [ -f "$f" ] || return 1
  local sz
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$sz" -gt 100 ] || return 1
  head -c 20 "$f" | grep -q "404: Not Found" && return 1
  return 0
}

fetch_one() {
  local fname="$1"
  local out="$DEST/$fname"
  # 1) 有 PAT -> 走 API
  if [ -n "$GITHUB_PAT" ]; then
    curl -sSL --max-time 60 -H "Authorization: token $GITHUB_PAT" \
      -H "Accept: application/vnd.github.raw" \
      -o "$out" "$API_BASE/$fname" 2>/dev/null
    if is_valid "$out"; then
      echo "  ok: $fname ($(stat -c%s "$out") bytes, API)"
      return 0
    fi
    rm -f "$out"
  fi
  # 2) 走当前活跃镜像
  if [ -n "$ACTIVE_MIRROR" ]; then
    curl -sSL --max-time 60 -o "$out" "$ACTIVE_MIRROR/$fname" 2>/dev/null
    if is_valid "$out"; then
      echo "  ok: $fname ($(stat -c%s "$out") bytes)"
      return 0
    fi
    rm -f "$out"
  fi
  return 1
}

echo "################################################"
echo "# 步骤 1/3：确保 federated/ 代码到位 -> $DEST"
echo "################################################"
mkdir -p "$DEST"

NEED=0
for f in "${FILES[@]}"; do
  is_valid "$DEST/$f" || NEED=1
done

if [ "$NEED" -eq 0 ]; then
  echo "[OK] 8 个文件均已就位且有效，跳过下载"
else
  echo "[..] 缺失或无效，开始下载"
  # 探测可用镜像 + 分支
  for br in main master; do
    for m in "https://raw.githubusercontent.com/$REPO/$br" \
             "https://raw.gitmirror.com/$REPO/$br" \
             "https://ghproxy.net/https://raw.githubusercontent.com/$REPO/$br"; do
      code=$(curl -sSL --max-time 20 -o /dev/null -w "%{http_code}" "$m/aggregate.py" 2>/dev/null)
      if [ "$code" = "200" ]; then
        ACTIVE_MIRROR="$m"
        echo "[..] 可用镜像: $m  (分支 $br)"
        break 2
      fi
    done
  done
  if [ -z "$ACTIVE_MIRROR" ] && [ -z "$GITHUB_PAT" ]; then
    echo "[!!] 所有镜像均不可用。可选："
    echo "     a) export GITHUB_PAT=<你的PAT> 后重跑"
    echo "     b) 本机 scp -r federated root@<host>:/root/HPT/"
    echo "     c) 本机打包 federated.tar.gz，用 JupyterLab 上传后解压到 $DEST"
    exit 1
  fi
  for f in "${FILES[@]}"; do
    fetch_one "$f" || echo "  [失败] $f"
  done
fi

MISSING=0
for f in "${FILES[@]}"; do
  is_valid "$DEST/$f" || { echo "  [缺] $f"; MISSING=1; }
done
if [ "$MISSING" -eq 1 ]; then
  echo
  echo "[!!] 关键文件仍不可用，请用上面提示的备用方式传输。"
  exit 1
fi
echo "[OK] federated/ 全部就位"
ls -la "$DEST/"

echo
echo "################################################"
echo "# 步骤 2/3：门0 —— 打印 model.pth 的 trunk 键名"
echo "################################################"
"$HPT_PY" - <<'PY'
import os, torch
from collections import Counter

p = "output/reach_baseline/model.pth"
if not os.path.exists(p):
    print(f"[!!] 找不到 {p}（需先完成集中式训练）")
    raise SystemExit(0)

sd = torch.load(p, map_location="cpu", weights_only=False)
print("state_dict 总键数:", len(sd))
print("顶层前缀分布:", Counter(k.split(".")[0] for k in sd).most_common(12))

tk = [k for k in sd if k.startswith("trunk")]
print()
print("trunk 键数量:", len(tk))
print("trunk 键示例:", tk[:6])

if not tk:
    print()
    print("[!!] 没有任何以 'trunk.' 开头的键！")
    print("     => aggregate.py 的 _is_trunk_key 会漏选，必须改判定逻辑。")
    print("     请把上面的『顶层前缀分布』贴给 AI 助手。")
else:
    inner = tk[0][len("trunk."):]
    print()
    print("[OK] trunk 键以 'trunk.' 开头 => aggregate.py 假设成立")
    print("     去掉首层 'trunk.' 后示例键:", inner)
    print("     (这正是写进 trunk.pth 的键名，供 policy.load_trunk 使用)")
PY

echo
echo "################################################"
echo "# 步骤 3/3：门1 —— 验证本地 trunk.pth 续训"
echo "################################################"
"$HPT_PY" "$DEST/verify_continuation.py"
echo
echo "门1 脚本退出码: $?"
echo
echo "================================================"
echo "若门0 显示 [OK] 且门1 输出 True -> 续训链路成立"
echo "下一步：开卡，改 fedavg_config.sh 为 NUM_ROUNDS=1 做门2 异构续训烟雾"
echo "================================================"
