#!/usr/bin/env bash
# Blocks edits to finished experiment records. Exit 2 blocks the action and sends
# stderr back to Claude as feedback. Exit 1 would NOT block -- that is the classic
# hooks footgun.
set -u
INPUT=$(cat)
FILE=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null)

case "$FILE" in
  */experiments/*/metrics.jsonl|*/experiments/*/train.log|*/experiments/*/scorecard.md)
    echo "Blocked: experiment records are append-only artifacts of a run. Create a new run instead of editing $FILE." >&2
    exit 2
    ;;
esac
exit 0
