#!/usr/bin/env bash
# FedVLA 云端自举：3 行命令完成「拉全部代码 + 门0 + 门1」
#
# 在 AutoDL 终端粘贴以下 3 行即可（不需要 git / 不需要 tar.gz / 不需要 scp）：
#
#   mkdir -p /root/HPT/federated && cd /root/HPT/federated
#   for f in cloud_gates.sh aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md; do curl -sSL --max-time 90 -O "https://raw.githubusercontent.com/civilizaa/FedVLA/main/federated/$f" || echo "fail $f"; done
#   bash /root/HPT/federated/cloud_gates.sh
#
# 说明：cloud_gates.sh 本身也会重新下载全部文件到 /root/HPT/federated/，
#       并用多镜像降级（raw -> gitmirror -> ghproxy），所以上面这步失败也没关系。
set -u

REPO="civilizaa/FedVLA"
BRANCH="main"
DEST="/root/HPT/federated"
MIRRORS=(
  "https://raw.githubusercontent.com/$REPO/$BRANCH/federated"
  "https://raw.gitmirror.com/$REPO/$BRANCH/federated"
  "https://ghproxy.net/https://raw.githubusercontent.com/$REPO/$BRANCH/federated"
  "https://gh-proxy.com/https://raw.githubusercontent.com/$REPO/$BRANCH/federated"
)
FILES=(cloud_gates.sh aggregate.py partition.py verify_continuation.py fedavg_config.sh run_fedavg.sh run_eval_fedavg.sh README.md)

mkdir -p "$DEST"
cd "$DEST" || exit 1

ok_one() {
  local f="$1"
  local sz
  sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
  [ "$sz" -gt 100 ] || return 1
  head -c 20 "$f" | grep -q "404: Not Found" && return 1
  return 0
}

ACTIVE=""
for m in "${MIRRORS[@]}"; do
  code=$(curl -sSL --max-time 20 -o /dev/null -w "%{http_code}" "$m/aggregate.py" 2>/dev/null)
  if [ "$code" = "200" ]; then
    ACTIVE="$m"
    echo "[bootstrap] 可用镜像: $m"
    break
  fi
done

if [ -z "$ACTIVE" ]; then
  echo "[bootstrap] 所有镜像不可用，请检查网络或改用 scp / JupyterLab 上传"
  exit 1
fi

for f in "${FILES[@]}"; do
  curl -sSL --max-time 90 -o "$f" "$ACTIVE/$f" 2>/dev/null
  if ok_one "$f"; then
    echo "  ok: $f ($(stat -c%s "$f") bytes)"
  else
    rm -f "$f"
    echo "  [失败] $f"
  fi
done

if [ ! -f "$DEST/cloud_gates.sh" ]; then
  echo "[bootstrap] 失败：cloud_gates.sh 未能下载"
  exit 1
fi

echo
echo "[bootstrap] 开始执行 cloud_gates.sh（内含门0 + 门1）"
echo "================================================"
bash "$DEST/cloud_gates.sh"
