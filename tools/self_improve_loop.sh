#!/usr/bin/env bash
# Layer 3 supervisor for the car_robot stack.
#
# This intentionally stays a plain bash loop. Each iteration gathers recent
# logs/images, launches Codex CLI in non-interactive mode, and validates that
# only the configured source allowlist changed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INTERVAL_SEC="${SELF_IMPROVE_INTERVAL_SEC:-180}"
MAX_ITERATIONS="${SELF_IMPROVE_MAX_ITERATIONS:-0}"
MODEL="${SELF_IMPROVE_MODEL:-${OMX_DEFAULT_FRONTIER_MODEL:-gpt-5.5}}"
RUN_ROOT="${SELF_IMPROVE_RUN_ROOT:-.car_self_improve}"
LOG_TAIL_LINES="${SELF_IMPROVE_LOG_TAIL_LINES:-220}"
MAX_LOGS="${SELF_IMPROVE_MAX_LOGS:-6}"
MAX_IMAGES="${SELF_IMPROVE_MAX_IMAGES:-6}"
CAPTURE_CURRENT="${SELF_IMPROVE_CAPTURE_CURRENT:-1}"

DEFAULT_ALLOWLIST=(
  "tools/reflex_drive.py"
  "tools/gemini_drive.py"
  "tools/vlm_local.py"
  "tools/depth_perception.py"
  "tools/self_improve_loop.sh"
  "docs/self_improvement.md"
  "README.md"
  ".env.example"
)

DEFAULT_LOG_GLOBS=(
  "logs/*.log"
  "logs/*.jsonl"
  ".omx/logs/*.jsonl"
  "${RUN_ROOT}/operator/*.log"
)

DEFAULT_IMAGE_GLOBS=(
  "logs/frames/*.jpg"
  "logs/frames/*.jpeg"
  "logs/frames/*.png"
  "*.jpg"
  "*.jpeg"
  "*.png"
  "logs/*.jpg"
  "logs/*.jpeg"
  "logs/*.png"
  "${RUN_ROOT}/operator/*.jpg"
  "${RUN_ROOT}/operator/*.jpeg"
  "${RUN_ROOT}/operator/*.png"
)

if [[ -n "${SELF_IMPROVE_ALLOWLIST:-}" ]]; then
  # Space-separated paths; keep names simple for an autonomous safety boundary.
  read -r -a ALLOWLIST <<< "$SELF_IMPROVE_ALLOWLIST"
else
  ALLOWLIST=("${DEFAULT_ALLOWLIST[@]}")
fi

if [[ -n "${SELF_IMPROVE_LOG_GLOBS:-}" ]]; then
  read -r -a LOG_GLOBS <<< "$SELF_IMPROVE_LOG_GLOBS"
else
  LOG_GLOBS=("${DEFAULT_LOG_GLOBS[@]}")
fi

if [[ -n "${SELF_IMPROVE_IMAGE_GLOBS:-}" ]]; then
  read -r -a IMAGE_GLOBS <<< "$SELF_IMPROVE_IMAGE_GLOBS"
else
  IMAGE_GLOBS=("${DEFAULT_IMAGE_GLOBS[@]}")
fi

die() {
  echo "self_improve_loop: $*" >&2
  exit 1
}

hash_file() {
  shasum -a 256 "$1" | awk '{print $1}'
}

mtime() {
  stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"
}

path_allowed() {
  local path="$1"
  local allowed
  for allowed in "${ALLOWLIST[@]}"; do
    [[ "$path" == "$allowed" ]] && return 0
  done
  return 1
}

snapshot_sources() {
  git ls-files -co --exclude-standard |
    sort |
    while IFS= read -r path; do
      [[ -f "$path" ]] || continue
      [[ "$path" == "$RUN_ROOT/"* ]] && continue
      printf '%s %s\n' "$(hash_file "$path")" "$path"
    done
}

latest_by_mtime() {
  local max_count="$1"
  shift
  local tmp="$1"
  shift
  : > "$tmp"

  local pattern path
  for pattern in "$@"; do
    while IFS= read -r path; do
      [[ -f "$path" ]] || continue
      [[ "$path" == "$RUN_ROOT/"* ]] && continue
      printf '%s\t%s\n' "$(mtime "$path")" "$path" >> "$tmp"
    done < <(compgen -G "$pattern" || true)
  done

  sort -rn "$tmp" | awk -F '\t' '!seen[$2]++ {print $2}' | head -n "$max_count"
}

load_car_ip_from_env_file() {
  [[ -n "${CAR_IP:-}" || ! -f .env ]] && return 0
  local value
  value="$(awk -F '=' '
    /^[[:space:]]*CAR_IP[[:space:]]*=/ {
      v=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
      gsub(/^["'\'']|["'\'']$/, "", v)
      print v
      exit
    }
  ' .env)"
  [[ -n "$value" ]] && export CAR_IP="$value"
}

write_context() {
  local run_dir="$1"
  local context="$2"
  shift 2
  local logs=("$@")

  {
    echo "# car_robot layer-3 self-improvement context"
    echo
    echo "Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Model: ${MODEL}"
    echo
    echo "## Architecture contract"
    echo "- Layer 1: fast real-time controller in tools/reflex_drive.py owns motor commands and safety."
    echo "- Layer 2: slow VLM goal/planning loop biases layer 1 and must not directly own continuous control."
    echo "- Layer 3: this Codex loop may tune algorithms/prompts/code from evidence, but must not drive motors."
    echo
    echo "## Editable allowlist"
    printf -- "- %s\n" "${ALLOWLIST[@]}"
    echo
    echo "## Evidence available this pass"
    echo "- Recent logs below include the layer-1 decision stream logs/reflex_*.jsonl: one JSON object per controller tick with per-zone openness (left/center/right), the chosen action, the VLM goal, and the exploration-map state."
    echo "- Attached images are annotated camera frames from logs/frames/ (newest first). Each overlay shows the same L/C/R openness, the action taken, the VLM goal, and the map state, so you can correlate what the car SAW with what it DID. Bars: red<block, orange<slow, yellow<open, green=open."
    echo "- The tests/ harness (tools, NOT in the allowlist) pins the controller's intent; you must not weaken or edit it — make it pass."
    echo
    echo "## Git status before iteration"
    git status --short || true
    echo
    echo "## Recent logs"
  } > "$context"

  local log
  if [[ "${#logs[@]}" -eq 0 ]]; then
    echo "No log files matched configured globs." >> "$context"
  else
    for log in "${logs[@]}"; do
      {
        echo
        echo "### ${log}"
        tail -n "$LOG_TAIL_LINES" "$log" || true
      } >> "$context"
    done
  fi

  {
    echo
    echo "## Required behavior"
    echo "- Prefer no-op plus a short report if the evidence does not justify a code change."
    echo "- Never print, persist, or rewrite secrets from .env."
    echo "- Do not edit firmware unless the allowlist explicitly includes that firmware path."
    echo "- Do not issue /control motor commands; fresh /capture reads are allowed."
    echo "- Keep changes small, reversible, and limited to the allowlist."
    echo "- Preserve strict road-roller target matching: require a visible roller drum."
    echo "- Do not infer a wall solely from low image-change."
    echo
    echo "## Required validation before final answer"
    echo "- Run: .venv/bin/python -m py_compile tools/reflex_drive.py tools/gemini_drive.py tools/vlm_local.py tools/depth_perception.py"
    echo "- Run: CAR_ROBOT_NO_VENV_REEXEC=1 .venv/bin/python -m unittest discover -s tests"
    echo "- The unittest harness must stay green. If a change legitimately alters expected behavior, report it; do NOT edit tests/ to force a pass (tests/ is outside the allowlist and a changed test file will block the pass)."
    echo "- Run: git diff --check"
    echo "- If full diff-check reports only pre-existing non-allowlisted files, also run git diff --check -- <edited allowlist files>."
    echo "- Report changed files, evidence used, validation output, pre-existing check gaps, and any no-op reason."
  } >> "$context"
}

validate_boundaries() {
  local before="$1"
  local after="$2"
  local report="$3"
  local paths="${report}.paths"
  local path before_value after_value

  : > "$report"
  { awk '{print $2}' "$before"; awk '{print $2}' "$after"; } | sort -u > "$paths"

  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if ! path_allowed "$path"; then
      before_value="$(awk -v p="$path" '$2 == p {print $1; exit}' "$before")"
      after_value="$(awk -v p="$path" '$2 == p {print $1; exit}' "$after")"
      if [[ "$before_value" != "$after_value" ]]; then
        echo "$path" >> "$report"
      fi
    fi
  done < "$paths"

  rm -f "$paths"
  sort -u "$report" -o "$report"
  [[ ! -s "$report" ]]
}

run_iteration() {
  local iteration="$1"
  local run_id run_dir context image_tmp log_tmp before after violation_report final_message codex_json
  run_id="$(date -u '+%Y%m%dT%H%M%SZ')-${iteration}"
  run_dir="${RUN_ROOT}/runs/${run_id}"
  mkdir -p "$run_dir"

  before="${run_dir}/before.sha256"
  after="${run_dir}/after.sha256"
  violation_report="${run_dir}/boundary_violations.txt"
  final_message="${run_dir}/codex_final.md"
  codex_json="${run_dir}/codex_events.jsonl"
  context="${run_dir}/context.md"
  image_tmp="${run_dir}/images.tmp"
  log_tmp="${run_dir}/logs.tmp"

  snapshot_sources > "$before"

  load_car_ip_from_env_file
  if [[ "$CAPTURE_CURRENT" == "1" && -n "${CAR_IP:-}" ]]; then
    local current_image="${run_dir}/current_capture.jpg"
    if curl -fsS -m 8 "http://${CAR_IP}/capture" -o "$current_image"; then
      if ! head -c 2 "$current_image" | LC_ALL=C grep -q $'\xff\xd8'; then
        rm -f "$current_image"
      fi
    else
      rm -f "$current_image"
    fi
  fi

  recent_logs=()
  while IFS= read -r item; do
    [[ -n "$item" ]] && recent_logs+=("$item")
  done < <(latest_by_mtime "$MAX_LOGS" "$log_tmp" "${LOG_GLOBS[@]}")

  recent_images=()
  while IFS= read -r item; do
    [[ -n "$item" ]] && recent_images+=("$item")
  done < <(latest_by_mtime "$MAX_IMAGES" "$image_tmp" "${IMAGE_GLOBS[@]}")
  if [[ -f "${run_dir}/current_capture.jpg" ]]; then
    recent_images=("${run_dir}/current_capture.jpg" "${recent_images[@]}")
  fi

  write_context "$run_dir" "$context" "${recent_logs[@]}"

  local codex_args=(
    exec
    --cd "$ROOT"
    --sandbox workspace-write
    --ask-for-approval never
    --model "$MODEL"
    --json
    --output-last-message "$final_message"
  )

  local image
  for image in "${recent_images[@]}"; do
    codex_args+=(--image "$image")
  done

  echo "[self-improve] iteration=${iteration} run_dir=${run_dir} model=${MODEL}"
  echo "[self-improve] logs=${#recent_logs[@]} images=${#recent_images[@]} interval=${INTERVAL_SEC}s"

  if ! codex "${codex_args[@]}" - < "$context" > "$codex_json"; then
    echo "[self-improve] codex exec failed; see ${codex_json}" >&2
    return 1
  fi

  snapshot_sources > "$after"
  if ! validate_boundaries "$before" "$after" "$violation_report"; then
    echo "[self-improve] blocked: Codex changed files outside allowlist:" >&2
    sed 's/^/  - /' "$violation_report" >&2
    echo "[self-improve] changes were left in place for manual inspection; no destructive revert was run" >&2
    return 2
  fi

  echo "[self-improve] done; final=${final_message}"
}

main() {
  command -v codex >/dev/null || die "codex CLI not found in PATH"
  mkdir -p "${RUN_ROOT}/runs"

  local iteration=0
  while true; do
    iteration=$((iteration + 1))
    run_iteration "$iteration"

    if [[ "$MAX_ITERATIONS" != "0" && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
      break
    fi
    sleep "$INTERVAL_SEC"
  done
}

main "$@"
