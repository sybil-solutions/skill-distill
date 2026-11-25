#!/usr/bin/env bash
# pi coding agent. MODEL like provider/id
set -uo pipefail
mkdir -p "$TRACE_DIR/sessions"
cd "$WORKSPACE" || exit 2

args=(-p --model "$MODEL" --session-dir "$TRACE_DIR/sessions" --no-skills)
if [[ "${PI_KEEP_CONTEXT:-0}" != "1" ]]; then
  args+=(--no-context-files)
fi
if [[ "${PI_KEEP_EXTENSIONS:-0}" != "1" ]]; then
  args+=(--no-extensions)
fi
if [[ -n "${STUDENT_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=($STUDENT_ARGS)
  args+=("${extra[@]}")
fi

pi "${args[@]}" -- "$(cat "$PROMPT_FILE")"
rc=$?
# shellcheck disable=SC2012
f=$(ls -t "$TRACE_DIR"/sessions/*.jsonl "$TRACE_DIR"/sessions/*/*.jsonl 2>/dev/null | head -1 || true)
[[ -n "$f" ]] && cp "$f" "$TRACE_DIR/session.jsonl"
exit "$rc"
