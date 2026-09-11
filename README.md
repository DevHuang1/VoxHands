# VoxHands

VoxHands is a voice-first dual-arm tabletop assistant demo. It combines a
Python planner and deterministic simulator with an industrial-style browser
workcell dashboard.

The demo includes:

- Structured natural-language command parsing.
- Safety validation before execution.
- Left/right arm assignment and parallel-safe table setting.
- A Three.js WebGL workcell with articulated arms, grippers, tabletop objects,
  targets, lighting, and camera views.
- Rapier3D browser physics at a fixed 60 Hz for contact, grasp, release, and
  settling telemetry.
- Timestamped minimum-jerk motion and a server-authoritative task status.
- Honest runtime labels for optional OpenVINO, Speechmatics, MuJoCo, and
  LeRobot integrations.

## Quick start

Requirements:

- Python 3.10 or newer.
- A modern browser with JavaScript and WebGL enabled.
- Network access to the pinned Three.js and Rapier3D CDN modules for the full
  3D workcell. The dashboard falls back to a dependency-free Canvas preview if
  those modules are unavailable.

The core MVP has no required third-party Python packages:

```bash
cd /Users/yuza/Desktop/VoxHands
python3 run.py
```

Open <http://127.0.0.1:8000/>. Keep the page on the HTTP server URL; opening
`frontend/index.html` directly only provides the offline preview and cannot
use the live API normally.

To stop the server, press `Ctrl+C` in the terminal where it is running.

## Try the demo

Use the default command or paste this into the command field:

```text
Set the table for two. Put the blue plate on the left and the cup on the right. Avoid the red zone.
```

The expected assignment is:

| Object | Arm | Destination |
| --- | --- | --- |
| Blue plate | Left | Left place |
| Cup | Right | Right place |

Useful controls in the dashboard:

- `Run` submits the command in the input field.
- `Run table set` runs the standard plate-and-cup scenario.
- `Home arms` sends a home-arms command.
- `Move plate` runs a single-object placement example.
- `Calibration` displays the calibration-status response.
- `Pause`, `Resume`, and `Stop` control an active simulation.
- `Reset` returns the server and browser workcell to the latest reset state.
- `Test contact` injects a local cup/barrier contact for telemetry testing.
- `Top`, `Operator`, and `Reset view` change the 3D camera.

The visible red item is the no-go safety barrier. It is not a grippable
object. A request such as `Pick up the red one and place it on the left.` is
blocked rather than silently converted into the default table-setting task.

## How the system works

```text
Browser command
      |
      v
POST /api/command
      |
      v
Python planner -> safety validator -> deterministic simulator
                                      |
                                      v
                              GET /api/state
                                      |
                                      v
                         Dashboard + browser physics
```

The Python simulator owns the plan, arm assignment, task phase, object
authority, and final status. The browser consumes that state, interpolates
the timestamped motion, and reports physics telemetry. Browser telemetry can
describe contact and settling, but it cannot change the server's safety
decision or task status.

### Browser state synchronization

Rapier object bodies are initialized from the latest server state, not from an
offline fallback snapshot. Each object tracks its authoritative run and pose.
When a reset, new run, plan, or server pose arrives, an unheld object is
reconciled for position, support height, rotation, velocity, and sleep state.
Actively held, releasing, and settling objects are protected from an
authoritative snap so the browser grasp/release handoff remains stable.

The wrist probe is a sensor, so it can participate in safety/contact
observation without physically pushing the cup away from the server pose.

## Motion and grasp contract

- Server motion is sampled at 60 Hz and uses minimum-jerk easing.
- Browser physics uses a fixed 60 Hz timestep.
- Normal placement tolerance is 1 cm at the target center.
- Grasp alignment requires at most 6 mm 3D wrist/object-center distance and at
  most 3 degrees of yaw error.
- Jaw-pad contact is held for four physics steps when available.
- If sensor contact is unavailable after the retry window, the dashboard uses
  the documented center-and-yaw fallback and displays `CONTACT FALLBACK`.
- Release follows `placing -> releasing -> settling -> returning -> parked`.
- Objects are target-locked only after the server reports the final placement
  and the browser has completed its settling handoff.

The browser physics path is a reliable simulator/demo contract, not a
hardware-calibrated robot controller or a MuJoCo execution. No hardware,
camera, NPU, or Speechmatics credentials are assumed by this repository.

## HTTP API

The local server exposes these endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Basic service health check. |
| `GET` | `/api/state` | Current plan, objects, arms, motion, physics, events, and runtime status. |
| `POST` | `/api/command` | Submit `{"text":"..."}` for planning and execution. |
| `POST` | `/api/control` | Submit `{"action":"pause"}`, `resume`, or `stop`. |
| `POST` | `/api/reset` | Reset the deterministic simulator and return the new state. |
| `POST` | `/api/physics-event` | Submit browser contact, collision, settling, or telemetry data. |

Example health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Example command request:

```bash
curl -X POST http://127.0.0.1:8000/api/command \
  -H 'Content-Type: application/json' \
  -d '{"text":"Set the table for two. Put the blue plate on the left and the cup on the right."}'
```

The API is local and intentionally simple. The server remains authoritative;
clients submit intent and browser observations only.

## Project layout

```text
run.py                    Local server entry point
voxhands/planner.py       Command parsing and arm/target assignment
voxhands/safety.py        Plan validation and red-zone safety rules
voxhands/simulation.py    Deterministic 60 Hz server-side task simulation
voxhands/server.py        Standard-library HTTP server and API routes
voxhands/models.py        Plan, action, and snapshot data models
voxhands/integrations.py  Optional runtime availability detection
frontend/index.html       Dashboard, Three.js scene, and Rapier3D physics
tests/test_core.py        Planner, safety, simulation, and API-contract tests
```

## Verification

Run the Python tests and static checks from the repository root:

```bash
python3 -m pytest -q
python3 -m compileall -q voxhands tests
node -e 'const fs=require("fs"); const html=fs.readFileSync("frontend/index.html","utf8"); const scripts=html.split("<script>")[1].split("</script>")[0]; new Function(scripts); console.log("frontend JavaScript syntax OK");'
git diff --check
```

For a live browser check, verify all of the following:

1. Reset, then run the table-setting command.
2. Confirm the plan shows `Cup`, `right arm`, and `Right place`.
3. During gripping, confirm the right arm reports `holding: cup` and
   `grasp_confirmed: true` in the server state. The browser should show either
   both pad contacts or `CONTACT FALLBACK`.
4. Reload while the objects are already placed and confirm the objects remain
   on their target positions.
5. Run again without pressing Reset.
6. Confirm both objects finish settled and target-locked within 1 cm, with no
   safety-barrier collision.

## Optional integrations

The core MVP does not install or require these packages. Install them only in
the matching hardware, credentials, or hackathon environment and pin the
versions there:

```text
openvino==2026.3.0
openvino-genai==2026.3.0
speechmatics-rt
mujoco
lerobot
```

The optional list is recorded in
[`requirements-optional.txt`](requirements-optional.txt). The current UI
labels unavailable integrations as demo/unavailable instead of claiming that
the laptop is connected to hardware or an NPU.

## Future integration boundary

The intended next step is to replace `TableSettingSimulation` with the
Intel-provided MuJoCo/LeRobot backend while preserving the existing plan,
safety, HTTP, and browser telemetry contracts. Free-form model output should
continue to produce intent only; it must not drive robot motors directly.
