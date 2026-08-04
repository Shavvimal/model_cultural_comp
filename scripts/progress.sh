#!/usr/bin/env bash
# Collection progress: overall percentage, one line per model-language cell, rate.
# Usage: scripts/progress.sh   (or: make progress)
cd "$(dirname "$0")/.." || exit 1

BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; CYAN=$'\033[36m'

shopt -s nullglob
files=(data/collection_2026/*.jsonl)
if [ ${#files[@]} -eq 0 ]; then echo "no data yet"; exit 0; fi

total=$(cat "${files[@]}" | wc -l | tr -d ' ')
target=18000
pct=$((100 * total / target))
filled=$((pct / 2)); empty=$((50 - filled))
bar="$(printf '#%.0s' $(seq 1 $((filled > 0 ? filled : 1))))$(printf '.%.0s' $(seq 1 $((empty > 0 ? empty : 1))))"

echo "${BOLD}${CYAN}=== 2026 collection ===${RESET}"
echo "${BOLD}overall: ${pct}%${RESET}  [${GREEN}${bar}${RESET}]  ${total} / ${target} records (en+zh)"
echo ""

for f in "${files[@]}"; do
  n=$(wc -l < "$f" | tr -d ' ')
  name=$(basename "$f" .jsonl)
  if [ "$n" -ge 500 ]; then colour=$GREEN
  elif [ "$n" -ge 100 ]; then colour=$YELLOW
  else colour=$RED; fi
  cellbar=$(printf '#%.0s' $(seq 1 $(((n / 25) > 0 ? (n / 25) : 1))))
  printf "%-32s ${colour}%4d/500${RESET} ${DIM}%s${RESET}\n" "$name" "$n" "$cellbar"
done

echo "${DIM}---${RESET}"
recent=$(.venv/bin/python -c "import glob,os,time; fs=glob.glob('data/collection_2026/*.jsonl'); print(sum(1 for f in fs if time.time()-os.path.getmtime(f)<120))" 2>/dev/null || echo 1)
if pgrep -f collect_cloud_2026 >/dev/null; then
  if [ "$recent" -gt 0 ]; then
    echo "collector: ${GREEN}RUNNING${RESET} (actively writing)"
  else
    echo "collector: ${YELLOW}RUNNING${RESET} (idle/throttled)"
  fi
else
  echo "collector: ${RED}NOT RUNNING${RESET}"
fi
echo "${DIM}--- last log lines:${RESET}"
tail -3 data/collection_2026_run.log 2>/dev/null
tail -3 data/collection_2026_zh_run.log 2>/dev/null
