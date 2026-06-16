# Three-layer autonomy

This repository now treats the car stack as three layers:

1. Fast real-time controller: `tools/reflex_drive.py` captures frames, runs local perception, owns motor commands, and stops on stale or unsafe local state.
2. Slow VLM planner: Gemini or local VLM updates high-level goals, target state, and route bias. It does not own continuous motor control.
3. Self-improvement loop: `tools/self_improve_loop.sh` periodically launches Codex CLI to inspect recent logs/images and tune the layer 1/2 code or prompts.

Layer 3 is intentionally a bounded code supervisor, not another driver. It must not send `/control` commands or modify firmware unless the allowlist is changed explicitly.

## Run

Start the driving process in one terminal and tee logs to a local ignored log file:

```bash
mkdir -p logs
.venv/bin/python tools/reflex_drive.py \
  --target "toy road roller / steamroller / asphalt compactor / игрушечный каток" \
  --debug 2>&1 | tee -a logs/reflex_drive.log
```

Besides the teed stdout, `reflex_drive.py` writes two machine- and agent-friendly
artifacts under `logs/` (gitignored) that layer 3 consumes directly:

- `logs/reflex_<timestamp>.jsonl` — one JSON object per controller tick: per-zone
  openness (`left`/`center`/`right`), the chosen `action`, `speed`/`trim`, the VLM
  `goal`, and the exploration `map` state. Picked up by the layer-3 log globs.
- `logs/frames/*.jpg` — annotated camera frames saved on events (blocked, dead-end,
  loop-escape, retreat, target-push, arrived) and on a heartbeat. Each overlay shows
  the same L/C/R openness bars, the action, the VLM goal, and the map state, so the
  self-improvement agent can correlate *what the car saw* with *what it did*. Picked
  up by the layer-3 image globs.

Toggle/tune with `--no-frame-log`, `--frame-log-sec` (heartbeat; `0` = events only),
`--frame-log-keep` (ring size), and `--no-annotate-frames` (save raw frames).

Start the self-improvement loop in another terminal:

```bash
SELF_IMPROVE_INTERVAL_SEC=180 tools/self_improve_loop.sh
```

Use a 2-5 minute cadence by setting `SELF_IMPROVE_INTERVAL_SEC` between `120` and `300`.

For a single supervised pass:

```bash
SELF_IMPROVE_MAX_ITERATIONS=1 tools/self_improve_loop.sh
```

## Tests

A fast, dependency-free `unittest` harness pins the *intent* of the layer-1
controller and the perception scoring, so a layer-3 (or human) edit that breaks
behavior is caught:

```bash
CAR_ROBOT_NO_VENV_REEXEC=1 .venv/bin/python -m unittest discover -s tests
```

It runs in milliseconds: the controller decision table (`command_from_state`,
steering points toward the open side, goal clamping, staleness fallback) and
`depth_perception.free_space` on synthetic depth maps. A slower end-to-end
regression against the real Depth Anything model on repo frames is opt-in:

```bash
CAR_ROBOT_RUN_MODEL_TESTS=1 CAR_ROBOT_NO_VENV_REEXEC=1 \
  .venv/bin/python -m unittest discover -s tests
```

`tests/` is intentionally **outside** the layer-3 allowlist, so the supervisor
must keep the suite green rather than weaken it.

## Controls

- `SELF_IMPROVE_MODEL`: Codex model, default `${OMX_DEFAULT_FRONTIER_MODEL:-gpt-5.5}`.
- `SELF_IMPROVE_INTERVAL_SEC`: seconds between passes, default `180`.
- `SELF_IMPROVE_MAX_ITERATIONS`: `0` means run forever.
- `SELF_IMPROVE_ALLOWLIST`: space-separated files Codex may change.
- `SELF_IMPROVE_LOG_GLOBS`: space-separated log globs.
- `SELF_IMPROVE_IMAGE_GLOBS`: space-separated image globs.
- `SELF_IMPROVE_CAPTURE_CURRENT`: `1` captures `http://$CAR_IP/capture` for each pass when `CAR_IP` is available.

Each pass writes artifacts under `.car_self_improve/runs/<timestamp>-<n>/`, including context, Codex JSON events, the final Codex message, and before/after source hashes.

## Guardrails

- Codex runs with `--sandbox workspace-write --ask-for-approval never`.
- The loop snapshots source-file hashes before and after each pass.
- If any non-allowlisted source path changes, the loop stops and reports the path. It does not run a destructive revert.
- Required checks in the Codex prompt are `py_compile` for the Python control stack, the `tests/` unittest harness, and `git diff --check`. If full diff-check reports only pre-existing non-allowlisted files, the pass must also run scoped diff-check for the edited allowlist files and report the gap.
- The unittest harness lives outside the allowlist, so a pass that edits `tests/` is blocked by the boundary check — the supervisor must satisfy the tests, not rewrite them.
- Layer 3 should make no-op reports when logs/images do not justify a code change.
