#!/usr/bin/env bash
# FedVLA 云端前置检查：一键拉代码 + 门0 + 门1
#
# 用法（在 AutoDL 的 ~/HPT 下执行）：
#     bash /root/HPT/federated/cloud_gates.sh
# 或者从本机 scp 上来后执行。
#
# 门0 / 门1 都是纯 CPU 操作，**可以在无卡模式跑**（省 ¥0.9/小时）。
# 门2（异构续训烟雾）必须开卡，见 run_fedavg.sh。
#
# 设计：脚本不依赖 git 能用——代码到位检测失败时会自动尝试从 raw.githubusercontent 补齐。
set -u

HPT_PY=/root/miniconda3/envs/hpt/bin/python
HPT_DIR=/root/HPT
FEDVLA_DIR=/root/FedVLA
RAW_BASE="https://raw.githubusercontent.com/civilizaa/FedVLA/master"
# 私有仓库：raw 链接对匿名请求一律 404。
# 若设置了 GITHUB_PAT，则用带 token 的 API 下载（更可靠）。
GITHUB_PAT="${GITHUB_PAT:-}"
API_BASE="https://api.github.com/repos/civilizaa/FedVLA/contents/federated"

export MUJOCO_GL=egl
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export PYTHONPATH="$HPT_DIR:${PYTHONPATH:-}"

cd "$HPT_DIR" || { echo "找不到 $HPT_DIR"; exit 1; }

# 判断文件是否有效（404 页面只有 14 字节，内容为 "404: Not Found"）
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
  local out="$FEDVLA_DIR/federated/$fname"
  if [ -n "$GITHUB_PAT" ]; then
    # 私有仓库：走 GitHub API，用 token 鉴权
    curl -sSL --max-time 90 -H "Authorization: token $GITHUB_PAT" \
      -H "Accept: application/vnd.github.raw" \
      -o "$out" "$API_BASE/$fname" 2>/dev/null
  else
    curl -sSL --max-time 90 -o "$out" "$RAW_BASE/$fname" 2>/dev/null
  fi
  if is_valid "$out"; then
    echo "  ok: $fname ($(stat -c%s "$out") bytes)"
    return 0
  fi
  rm -f "$out"
  return 1
}

echo "################################################"
echo "# 步骤 1/3：确保 federated/ 代码到位"
echo "################################################"
mkdir -p "$FEDVLA_DIR/federated"
NEED=0
for f in aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md cloud_gates.sh; do
  is_valid "$FEDVLA_DIR/federated/$f" || NEED=1
done

if [ "$NEED" -eq 0 ]; then
  echo "[OK] federated/ 8 个文件均已就位且有效，跳过下载"
else
  echo "[..] 部分文件缺失或无效（404 页面），尝试下载"
  if [ -z "$GITHUB_PAT" ]; then
    echo "[..] 未设置 GITHUB_PAT：私有仓库匿名下载会返回 404（14 字节假文件）"
    echo "     方案A: export GITHUB_PAT=<你的PAT> 后重跑本脚本"
    echo "     方案B: 本机 Windows 打包 federated.tar.gz，用 JupyterLab 上传后解压（见 README）"
    echo "     方案C: 本机 scp -r federated root@<host>:/root/HPT/"
  fi
  for f in aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md cloud_gates.sh; do
    fetch_one "$f" || echo "  [跳过] $f 下载失败"
  done
fi

# 校验
MISSING=0
for f in aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh cloud_gates.sh; do
  is_valid "$FEDVLA_DIR/federated/$f" || { echo "  [缺] $f"; MISSING=1; }
done
if [ "$MISSING" -eq 1 ]; then
  echo
  echo "[!!] 关键文件仍不可用。请用上面【方案B】(JupyterLab 上传 tar.gz) 或【方案C】(scp)。"
  echo "     注意：若之前下载出 14 字节的假文件，请先 rm -rf $FEDVLA_DIR/federated 再操作。"
  exit 1
fi
echo "[OK] federated/ 全部就位"
ls -la "$FEDVLA_DIR/federated/"

echo
echo "################################################"
echo "# 步骤 2/3：门0 —— 打印 model.pth 的 trunk 键名"
echo "################################################"
"$HPT_PY" - <<'PY'
import os, torch
from collections import Counter

p = "output/reach_baseline/model.pth"
if not os.path.exists(p):
    print(f"[!!] 找不到 {p}（需先完成门1的集中式训练）")
    raise SystemExit(0)

sd = torch.load(p, map_location="cpu", weights_only=False)
print("state_dict 总键数:", len(sd))
print("顶层前缀分布:", Counter(k.split(".")[0] for k in sd).most_common(12))

tk = [k for k in sd if k.startswith("trunk")]
print()
print("trunk 键数量:", len(tk))
print("trunk 键示例:", tk[:6])

# 判定 aggregate.py 的假设是否成立
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
"$HPT_PY" "$FEDVLA_DIR/federated/verify_continuation.py"
echo
echo "门1 脚本退出码: $?"
echo
echo "================================================"
echo "若门0 显示 [OK] 且门1 输出 True -> 续训链路成立"
echo "下一步：开卡，改 fedavg_config.sh 为 NUM_ROUNDS=1 做门2 异构续训烟雾"
echo "================================================"
