#!/usr/bin/env bash
# Layer-0 evolution supervisor for the car_robot stack.
#
# Closes the autonomy loop: drive a bounded EPISODE, then run ONE layer-3
# self-improvement (reflection) pass, then start a FRESH driver so the new code
# takes effect, and repeat — forever.
#
# Why a separate supervisor: reflex_drive.py loads the source once at startup, so
# edits Codex makes do NOT take effect until the driver is restarted. This loop
# restarts the driver every cycle, which is what turns "edit the code" into
# "the robot actually behaves differently next time".
#
# Safety model (this script owns it; layer 3 still never drives):
#   - Each episode is time-boxed via reflex_drive.py --secs and self-exits.
#   - The car is explicitly stopped after every episode and on shutdown.
#   - Before each reflection pass the editable (allowlist) files are backed up.
#   - After the pass the tests/ harness is the AUTHORITATIVE gate: red tests or a
#     failed Codex pass -> the allowlist files are restored from backup (the bad
#     self-edit is rolled back). An out-of-allowlist edit halts the whole loop.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
EPISODE_SEC="${EVOLVE_EPISODE_SEC:-60}"          # driving seconds per episode
MAX_ITERATIONS="${EVOLVE_MAX_ITERATIONS:-0}"     # 0 = run forever
DRY_RUN="${EVOLVE_DRY_RUN:-0}"                   # 0 = real motors (default), 1 = --dry-run
REFLECT="${EVOLVE_REFLECT:-1}"                   # 0 = episodes only (no Codex; for testing)
COOLDOWN_SEC="${EVOLVE_COOLDOWN_SEC:-0}"         # pause between cycles
TARGET="${EVOLVE_TARGET:-yellow rubber bath duck / toy duckie / жёлтая резиновая уточка для ванной}"
RUN_ROOT="${EVOLVE_RUN_ROOT:-.car_evolve}"
LOG_DIR="${EVOLVE_LOG_DIR:-logs}"
STOP_FILE="${RUN_ROOT}/STOP"

# Editable files for the reflection pass. Single source of truth: exported so the
# layer-3 pass uses the exact same boundary (keeps backup/rollback in sync with it).
ALLOWLIST=(
  "tools/reflex_drive.py"
  "tools/gemini_drive.py"
  "tools/vlm_local.py"
  "tools/depth_perception.py"
  "tools/self_improve_loop.sh"
  "docs/self_improvement.md"
  "README.md"
  ".env.example"
)
export SELF_IMPROVE_ALLOWLIST="${ALLOWLIST[*]}"

TEST_CMD=(env CAR_ROBOT_NO_VENV_REEXEC=1 "$PY" -m unittest discover -s tests)

log()  { echo "[evolve] $*"; }
die()  { echo "[evolve] FATAL: $*" >&2; exit 1; }

load_car_ip() {
  [[ -n "${CAR_IP:-}" ]] && return 0
  [[ -f .env ]] || return 0
  local v
  v="$(awk -F= '/^[[:space:]]*CAR_IP[[:space:]]*=/{gsub(/^[ \t"'\'']+|[ \t"'\'']+$/,"",$2); print $2; exit}' .env)"
  [[ -n "$v" ]] && export CAR_IP="$v" || true
}

stop_car() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  load_car_ip
  [[ -z "${CAR_IP:-}" ]] && return 0
  curl -fsS -m 5 "http://${CAR_IP}/control?var=car&val=5" >/dev/null 2>&1 || true
}

DRIVER_PID=""
_CLEANED=0
cleanup() {
  [[ "$_CLEANED" == "1" ]] && return 0
  _CLEANED=1
  log "shutting down…"
  if [[ -n "$DRIVER_PID" ]] && kill -0 "$DRIVER_PID" 2>/dev/null; then
    # SIGINT: reflex_drive handles KeyboardInterrupt -> stops motors + closes logs.
    kill -INT "$DRIVER_PID" 2>/dev/null || true
    wait "$DRIVER_PID" 2>/dev/null || true
  fi
  stop_car
}
trap 'cleanup; exit 0' INT TERM
trap 'cleanup' EXIT

run_episode() {
  local n="$1"
  local elog="${LOG_DIR}/evolve_episode_${n}.log"
  local args=(--target "$TARGET" --secs "$EPISODE_SEC" --debug)
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)

  log "episode ${n}: driving ${EPISODE_SEC}s (dry_run=${DRY_RUN}) -> ${elog}"
  "$PY" tools/reflex_drive.py "${args[@]}" >"$elog" 2>&1 &
  DRIVER_PID=$!
  local rc=0
  wait "$DRIVER_PID" || rc=$?
  DRIVER_PID=""
  stop_car   # belt-and-suspenders halt even if the driver exited uncleanly
  log "episode ${n}: driver exited rc=${rc}"
}

backup_allowlist() {
  local dst="$1"; rm -rf "$dst"; mkdir -p "$dst"
  local f
  for f in "${ALLOWLIST[@]}"; do
    [[ -f "$f" ]] || continue
    mkdir -p "$dst/$(dirname "$f")"
    cp -p "$f" "$dst/$f"
  done
}

restore_allowlist() {
  local src="$1"; local f
  for f in "${ALLOWLIST[@]}"; do
    [[ -f "$src/$f" ]] || continue
    mkdir -p "$(dirname "$f")"
    cp -p "$src/$f" "$f"
  done
}

run_reflection() {
  local n="$1"
  local backup="${RUN_ROOT}/backup/${n}"
  backup_allowlist "$backup"

  log "reflection ${n}: running one self-improve pass…"
  local sc=0
  SELF_IMPROVE_MAX_ITERATIONS=1 tools/self_improve_loop.sh || sc=$?

  if [[ "$sc" -eq 2 ]]; then
    log "reflection ${n}: BOUNDARY VIOLATION — Codex edited a file outside the allowlist."
    die "halting for manual inspection (iteration ${n}); see .car_self_improve/runs and 'git status'."
  fi
  if [[ "$sc" -ne 0 ]]; then
    log "reflection ${n}: Codex pass failed (rc=${sc}); restoring allowlist, continuing."
    restore_allowlist "$backup"
    return 0
  fi

  log "reflection ${n}: running tests (authoritative gate)…"
  if "${TEST_CMD[@]}" >"${RUN_ROOT}/tests_${n}.log" 2>&1; then
    log "reflection ${n}: ACCEPTED (tests green) — new code takes effect next episode."
  else
    log "reflection ${n}: REJECTED (tests red) — rolling back self-edit."
    restore_allowlist "$backup"
    if "${TEST_CMD[@]}" >"${RUN_ROOT}/tests_${n}_postrollback.log" 2>&1; then
      log "reflection ${n}: rollback verified (tests green again)."
    else
      log "reflection ${n}: WARNING — tests still red after rollback; inspect ${RUN_ROOT}/tests_${n}_postrollback.log"
    fi
  fi
}

main() {
  [[ "$REFLECT" == "1" ]] && { command -v codex >/dev/null || die "codex CLI not found in PATH"; }
  mkdir -p "$RUN_ROOT/backup" "$LOG_DIR"
  rm -f "$STOP_FILE"

  log "evolution loop: episode=${EPISODE_SEC}s dry_run=${DRY_RUN} reflect=${REFLECT} max_iter=${MAX_ITERATIONS}"
  log "stop anytime: Ctrl-C, or 'touch ${STOP_FILE}'"

  local n=0
  while true; do
    [[ -f "$STOP_FILE" ]] && { log "STOP file present; exiting."; break; }
    n=$((n + 1))

    run_episode "$n"

    [[ -f "$STOP_FILE" ]] && { log "STOP file present after episode; exiting."; break; }

    if [[ "$REFLECT" == "1" ]]; then
      run_reflection "$n"
    else
      log "reflection ${n}: skipped (EVOLVE_REFLECT=0)."
    fi

    if [[ "$MAX_ITERATIONS" != "0" && "$n" -ge "$MAX_ITERATIONS" ]]; then
      log "reached MAX_ITERATIONS=${MAX_ITERATIONS}; exiting."
      break
    fi
    if [[ "$COOLDOWN_SEC" -gt 0 ]]; then
      log "cooldown ${COOLDOWN_SEC}s…"; sleep "$COOLDOWN_SEC"
    fi
  done
  log "evolution loop done after ${n} iteration(s)."
}

main "$@"
