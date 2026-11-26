#!/usr/bin/env bash
# opencode. MODEL like provider/model from your opencode config.
set -euo pipefail
cd "$WORKSPACE"
opencode run --format json --auto ${MODEL:+-m "$MODEL"} ${STUDENT_ARGS:-} "$(cat "$PROMPT_FILE")" \
  </dev/null | tee "$TRACE_DIR/session.jsonl"
# Some provider failures are emitted as JSON even when the process exits zero.
python3 - "$TRACE_DIR/session.jsonl" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as trace:
    for line in trace:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "error":
            sys.exit(1)
PY
