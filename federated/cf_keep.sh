#!/usr/bin/env bash
# ==============================================================================
# 抢救备份：把各配置的 r49 权重（全局 + 客户端本地）复制到 /root/fedcore/keep/
#
# 为什么必须
# ----------
# run_comm_scan_v2.sh 在评估结束后会做磁盘回收：
#     rm -rf $root/global_r*  $root/client_*_r*
# ⇒ 评估一结束，**干净参照（客户端本地权重）与量化后的全局模型一起被删掉**。
#   c_int8 就是这么丢的（目录已空）⇒ 归因取证失去对照组。
# 本脚本在回收之前把 r49 抢救出来（并等文件写完再拷，避免半截文件）。
#
# 用法
# ----
#   TAGS="c_bit1" bash /root/fedcore/cf_keep.sh
#   # 后台：nohup setsid bash /root/fedcore/cf_keep.sh > /dev/null 2>&1 < /dev/null &
# ==============================================================================
set -u

OUT=${OUT:-/root/HPT/output}
K=${K:-/root/fedcore/keep}
LOG=${LOG:-/root/HPT/output/cf_keep.log}
TAGS=${TAGS:-"c_bit1"}
TIMEOUT=${TIMEOUT:-14400}          # 最长等 4 小时
N_CLIENTS=${N_CLIENTS:-4}

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mkdir -p "$K"
log "===== 守望启动：tags=[$TAGS] out=$OUT keep=$K timeout=${TIMEOUT}s ====="

# 文件大小连续两次一致 ⇒ 认为写完（避免拷到半截文件）
stable() {
    local f=$1 s1 s2
    [ -f "$f" ] || return 1
    s1=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$s1" != "0" ] || return 1
    sleep 15
    s2=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$s1" = "$s2" ]
}

try_copy() {  # $1=源目录 $2=目标目录
    local src=$1 dst=$2
    [ -d "$src" ] || return 1
    [ -d "$dst" ] && return 0
    stable "$src/model.pth" || return 1
    cp -r "$src" "$dst" || return 1
    log "已备份 $src → $dst（$(du -sh "$dst" 2>/dev/null | cut -f1)）"
    return 0
}

t0=$(date +%s)
while : ; do
    pending=0
    for tag in $TAGS; do
        # 客户端本地权重（干净 normalizer 来源）——它们比 global_r49 先写完
        for c in $(seq 0 $((N_CLIENTS - 1))); do
            src=$OUT/$tag/client_${c}_r49
            dst=$K/${tag}_client${c}_r49
            if [ -d "$dst" ]; then continue; fi
            try_copy "$src" "$dst" || pending=$((pending + 1))
        done
        # 量化后的全局模型
        src=$OUT/$tag/global_r49
        dst=$K/${tag}_global_r49
        if [ ! -d "$dst" ]; then
            try_copy "$src" "$dst" || pending=$((pending + 1))
        fi
    done
    [ "$pending" -eq 0 ] && { log "全部备份完成"; break; }
    now=$(date +%s)
    if [ $((now - t0)) -ge "$TIMEOUT" ]; then
        log "超时（还有 $pending 项未备份）"
        break
    fi
    sleep 60
done
log "===== 守望退出 ====="
