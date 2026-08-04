#!/usr/bin/env bash
# Collection progress: one line per model-language cell, plus rate.
# Usage: scripts/progress.sh   (or: make progress)
cd "$(dirname "$0")/.." || exit 1

echo "=== 2026 collection progress (target 500/cell) ==="
for f in data/collection_2026/*.jsonl; do
  [ -e "$f" ] || { echo "no data yet"; exit 0; }
  n=$(wc -l < "$f")
  name=$(basename "$f" .jsonl)
  bar=$(printf '#%.0s' $(seq 1 $((n / 25))))
  printf "%-32s %4d/500 %s\n" "$name" "$n" "$bar"
done

total=$(cat data/collection_2026/*.jsonl | wc -l)
echo "---"
echo "total records: $total / 18000 (en+zh)"

recent=$(find data/collection_2026 -name '*.jsonl' -newermt '-2 minutes' | wc -l)
if pgrep -f collect_cloud_2026 >/dev/null; then
  echo "collector: RUNNING ($( [ "$recent" -gt 0 ] && echo 'actively writing' || echo 'idle/throttled' ))"
else
  echo "collector: NOT RUNNING"
fi
echo "--- last log lines:"
tail -3 data/collection_2026_run.log 2>/dev/null
tail -3 data/collection_2026_zh_run.log 2>/dev/null
