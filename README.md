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
- Three.js and Rapier3D are bundled in `frontend/vendor` with their licenses;
  the 3D workcell requires no CDN access. Canvas remains the WebGL fallback.

The core MVP has no required third-party Python packages:

```bash
cd E:\hackathon\VoxHands
python run.py
```

Open <http://127.0.0.1:8000/>. Keep the page on the HTTP server URL; opening
`frontend/index.html` directly only provides the offline preview and cannot
use the live API normally.

To stop the server, press `Ctrl+C` in the terminal where it is running.

## Deploy to Render

Render is a good fit because VoxHands is a stateful, long-running process:
motion runs on a background thread and the browser polls `/api/state`. A single
Render web service serves both the dashboard and the API (do **not** use
Gunicorn/multiple workers; they would break the shared simulation state).

Deploy with the included blueprint:

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** (`render.yaml` is included).
3. Supply the `GROQ_API_KEY` env var when prompted (leave blank for keyword
   planning), or add it in the service's **Environment** tab afterward.
4. Keep `HOST`/`PORT` defaults: the server binds `0.0.0.0` and reads Render's
   `$PORT`. Health checks use `/api/health`.

The server honors `Ctrl+C` locally and handles Render's `SIGTERM` gracefully.
Free-tier instances can sleep after idle; a cold start can take about a minute
before the first request wakes the service.

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

## Groq AI conversational control (optional)

Set a Groq API key and the command pipeline upgrades from the keyword planner to
a full conversational controller:

```bash
# option A: shell environment variable
export GROQ_API_KEY="gsk_..."

# option B: project .env.local (auto-loaded, git-ignored)
cp .env.local.example .env.local
# then paste your key into .env.local  ->  GROQ_API_KEY=gsk_...

python run.py
```

No Python package is needed: the client talks to Groq's OpenAI-compatible
endpoint with the standard library only. When the key is absent, commands fall
back to the deterministic keyword planner and the dashboard shows
`Groq AI · unavailable`. `GROQ_MODEL` may also be set in the shell or
`.env.local` to override the default model `openai/gpt-oss-120b`. If the
primary request fails, VoxHands tries `GROQ_FALLBACK_MODEL` (default
`openai/gpt-oss-20b`) before falling back to the deterministic keyword planner.

The AI understands natural language motion intent, including style cues and
gestures:

- **Movement styles** (speed, easing, and carry trajectory):
  `standard`, `gentle` (slow/soft), `precise` (careful/low), `rapid` (fast),
  `playful` (bouncy arc), `wavy` (sinuous weave).
- **Gestures** performed with both hands after placement: `wave`, `bow`,
  `dance`, `point`.
- **Conditions**: hold/pause at the target (`pause_ms`) and sequencing notes
  such as "after the plate is placed" or "only if the path is clear" are carried
  on the action and reflected in the plan; the red-zone, occupancy, and
  serialization guards are always enforced by the safety validator.
- **Conversational turns**: greetings and questions (`"hello"`, `"what can you
  do?"`) get an AI reply without starting motion. The reply appears in the
  command panel; a chat-only alias is `POST /api/chat`.

The dashboard adds **Style** and **Gesture** selectors that are sent with every
command as JSON overrides, so a styled run works even with the keyword fallback.
An example styled request:

```text
gently place the cup on the right and the plate on the left, then wave
```

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
the timestamped motion, and reports physics telemetry. Browser telemetry describes contact and settling. A collision report carrying
the current plan ID triggers a server protective stop during execution. Other
telemetry cannot set task status; reset is required after stopping.

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
| `POST` | `/api/plan` | Validate and preview a command without starting a run. |
| `POST` | `/api/command` | Submit `{"text":"...", "style?":"", "gesture?":""}` for planning and execution. Conversational intents return an AI reply without starting motion. |
| `POST` | `/api/chat` | Submit `{"text":"..."}` and get an AI reply only; never starts a motion run. |
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

Example read-only plan preview:

```bash
curl -X POST http://127.0.0.1:8000/api/plan \
  -H 'Content-Type: application/json' \
  -d '{"text":"Place the cup on the right."}'
```

The preview returns `valid`, `safety_issues`, and the structured plan. It does
not alter the current run, metrics, object positions, or event log.

The API is local and intentionally simple. The server remains authoritative;
clients submit intent and browser observations only.

## Project layout

```text
run.py                    Local server entry point (reads HOST/PORT env)
render.yaml               Render Blueprint deployment configuration
runtime.txt               Render Python version pin (3.11.9)
requirements.txt          Empty on purpose; VoxHands is standard-library-only
voxhands/planner.py       Keyword command parsing and arm/target assignment
voxhands/groq.py          Groq LLM client, system prompt, and AI plan builder
voxhands/styles.py        Movement-style registry, easing, and gesture keyframes
voxhands/safety.py        Plan validation and red-zone safety rules
voxhands/simulation.py    Deterministic 60 Hz server-side task simulation
voxhands/server.py        Standard-library HTTP server and API routes
voxhands/models.py        Plan, action, and snapshot data models
voxhands/integrations.py  Optional runtime availability detection
frontend/index.html       Dashboard, Three.js scene, and Rapier3D physics
tests/                    Keyword, AI, simulation, and API-contract tests
```

## Verification

Run the Python tests and static checks from the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q voxhands tests
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
safety, HTTP, and browser telemetry contracts. Groq conversational planning is
implemented and gated on `GROQ_API_KEY`; free-form model output still produces
intent and action structure only, never raw motor commands.

## Improved command and execution behavior

Commands bind a destination to each named object, including reversed sides,
center/centre/middle, and cup/mug aliases. Example:
`Place the cup on the left and the plate on the right.`
Missing destinations, conflicting destinations, unsupported wording, and negation
produce clarification messages; the simulator does not execute a partial guess.
The explicit table-setting command supplies the standard pair automatically.

Home arms parks idle simulated arms while preserving objects. Calibration reports
that no physical camera/hardware calibration exists. New placement requests are
rejected during running/paused/stopped states; finish or reset first.

The standard separated plate/cup lanes run together. Other moves execute in order,
with longer durations for barrier detours. Paths check object-center segments
against a padded 2D barrier rectangle and detour through the front of the table.
This is NOT full-link collision detection, object-shape clearance, calibrated
inverse kinematics, or a safety-certified hardware controller. Browser collision
reports for the current run stop motion; stale run IDs cannot stop a new run.

Planning latency is measured. Perception and placement telemetry explicitly refer
to simulation. OpenVINO and Speechmatics package detection does not mean an adapter
is active. Real model inference, Speechmatics streaming, MuJoCo/LeRobot execution,
and physical calibration remain future integration work requiring a selected
model/backend, credentials where applicable, and appropriate hardware testing.
Browser microphone support depends on browser permissions and speech support;
typed commands remain available. Browser speech may require an internet connection.

Vendored modules: Three.js 0.164.1 (MIT), Rapier3D compat 0.12.0 (Apache-2.0).
Their original license files are beside the modules in `frontend/vendor`.

## Command assistance

Use the Object and Destination selectors, then **Use command** to fill the command
field; **Run** executes it. A request such as `move spoon` keeps Spoon in the
recognized-object list and offers complete destination commands beside the input.
Clarification does not start motion or lower the execution success rate.

Requests to place an object at another object's occupied target center are blocked
with a message naming that object. This occupancy check complements the limited
center-path checks described above; it does not replace shape/link collision tests.
Execution success rate counts started runs, not requests needing clarification.
Speech controls are disabled when the browser exposes no speech-recognition API;
browser permission and actual recognition service availability still vary.

Container actions are supported explicitly: `Put the spoon into the cup` creates
one `spoon -> cup_interior` action, tracks the cup's live position, lowers the
spoon above the cup base, and records `container: "cup"` only after release.
The cup itself cannot be selected as its own container destination.
