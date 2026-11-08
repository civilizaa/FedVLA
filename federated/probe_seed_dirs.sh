#!/usr/bin/env bash
# 检查 hydra multirun 每个 seed 子目录的完成情况。
# 用法：bash probe_seed_dirs.sh <评估输出目录>
set -u
D="${1:?}"
for d in "$D"/*/; do
  [ -d "$d" ] || continue
  f="$d/run_eval.log"
  if [ -f "$f" ]; then
    n=$(grep -c "| avg" "$f" 2>/dev/null || true)
    [ -n "$n" ] || n=0
    echo "$(basename "$d") tables=$n size=$(stat -c %s "$f")"
  else
    echo "$(basename "$d") NO_LOG"
  fi
done
