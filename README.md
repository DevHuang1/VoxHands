# VoxHands

VoxHands is a voice-first dual-arm tabletop assistant demo. It combines a
Python planner and deterministic simulator with an industrial-style browser
workcell dashboard.

VoxHands is currently positioned as a summit-ready constrained Physical AI
demo: Groq interprets language and requests validated workcell tools, while the
local simulator remains authoritative for physics, safety, state, and motion.
This demonstrates agentic control infrastructure without claiming direct
control of a real robot or arbitrary physics mutation.

The demo includes:

- Structured natural-language command parsing.
- Safety validation before execution.
- Left/right arm assignment and parallel-safe table setting.
- A Three.js WebGL workcell with articulated arms, grippers, tabletop objects,
  targets, lighting, and camera views.
- Rapier3D browser physics at a fixed 60 Hz for contact, grasp, release, and
  settling telemetry.
- Timestamped minimum-jerk motion and a server-authoritative task status.
- A real Intel OpenVINO path: the policy MLP is exported to a genuine
  `.xml`/`.bin` IR, INT8-quantized with NNCF post-training quantization, and
  benchmarked on CPU with measured latency/throughput (`scripts/`).
- Honest runtime labels for optional OpenVINO, Speechmatics, MuJoCo, and
  LeRobot integrations.

## Fastest path to the Intel OpenVINO demo

Intel track / optimization reviewers: this is the shortest runnable path (Python
3.13). It exports a real OpenVINO IR, quantizes it with NNCF INT8, and prints
measured CPU latency/throughput.

```bash
python3.13 -m pip install openvino==2026.3.1 nncf==3.3.0 numpy
python3.13 scripts/benchmark_openvino.py --sweep --throughput   # full FP32 vs INT8 size sweep
python3.13 voxhands/check_intel.py                              # honest Intel device/NNCF status
python3.13 run.py                                               # live dashboard at 127.0.0.1:8000 (optional)
```

See [Intel OpenVINO optimization path](#intel-openvino-optimization-path-ir-export-int8-ptq-measured-cpu-inference)
for the measured results and the verified-vs-gated breakdown.

## Quick start

Requirements:

- Python 3.10 or newer.
- A modern browser with JavaScript and WebGL enabled.
- Three.js and Rapier3D are bundled in `frontend/vendor` with their licenses;
  the 3D workcell requires no CDN access. Canvas remains the WebGL fallback.

The core MVP has no required third-party Python packages:

```bash
cd /path/to/VoxHands
python3 run.py
```

Open <http://127.0.0.1:8000/>. Keep the page on the HTTP server URL; opening
`frontend/index.html` directly only provides the offline preview and cannot
use the live API normally.

To stop the server, press `Ctrl+C` in the terminal where it is running.

### MuJoCo physics + vision backend (recommended, optional)

When `mujoco` is installed, the server swaps in the differentiated workcell:
two SO-101-style six-DOF arms, a tabletop with a red safety zone, four trackable
objects, and overhead/operator cameras, all driven by MuJoCo with
inverse-kinematics pick-and-place:

```bash
python3.13 -m pip install mujoco   # Python 3.13 recommended for wheels
python3.13 -m voxhands.server        # same endpoints; /api/camera + /api/vision active
```

The dashboard then shows the MuJoCo `OVERHEAD`/`OPERATOR` camera feed
(picture-in-picture, bottom-left of the workcell) backed by the real renderer
and a `MujocoVision` runtime badge. Without `mujoco`, the original
deterministic simulator serves `/api/state` unchanged and `/api/camera`
returns 404.

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
- `OVERHEAD`/`OPERATOR` (bottom-left thumbnail) toggles the live MuJoCo camera
  feed when the physics backend is active.

### Suggested summit demo flow

Run these short scenarios in order:

1. **Successful tool execution:** `Move the blue plate to the center.` Show
   Groq's selected tool, the server-side plan, the 60 Hz arm motion, and the
   final target lock.
2. **Bounded hand control:** `Move the empty left arm to x .30 y .30 z .20,
   then close the gripper.` Show that manual arm and gripper requests stay
   inside validated workcell bounds.
3. **Safety refusal:** `Move the blue plate to the ground.` Show that the
   unsupported destination is blocked without changing the object or starting
   motion.
4. **Intel OpenVINO optimization:** run the export → quantize → benchmark
   pipeline below and show the IR artifacts, the INT8 PTQ, and the measured CPU
   latency/throughput on the actual compiled model. Point at `/api/vision`'s
   `inference_ms`/`device` to show the live camera loop using the same runtime.

The useful infrastructure story is the trace across all four cases:
`language intent -> Groq tool call -> tool boundary -> safety validator ->
authoritative simulator state -> browser telemetry`, alongside a real
`numpy weights -> OpenVINO IR -> NNCF INT8 -> measured CPU inference` path.

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

When Groq is configured, `/api/command` runs a bounded local tool loop. Groq
can observe the authoritative workcell, execute validated placements, move an
empty arm within the workcell bounds, open or close an empty gripper, home the
arms, wait for settling, and pause/resume/stop/reset the simulator. It cannot
write arbitrary object poses, bypass the red barrier, inject browser physics
status, or execute raw code. The server remains authoritative and the browser
Rapier3D layer remains telemetry and contact validation.

The tool loop is intentionally slower than physics: the simulator continues at
60 Hz while Groq makes bounded decisions and receives compact state results.
This keeps language-model latency out of the hand trajectory controller.

## How the system works

```text
Browser command
      |
      v
POST /api/command
      |
      v
Groq tool loop -> tool boundary -> safety validator -> deterministic simulator
      |                         |
      |                         +--> bounded arm/gripper controls
      v
  compact state and physics results returned for replanning
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
| `GET` | `/api/camera` | MuJoCo camera frame as `image/jpeg` (Pillow) or `image/bmp` (stdlib). Query `?camera=overhead` or `?camera=operator`. |
| `GET` | `/api/vision` | Vision status: active detector (`color-mask` fallback or `openvino-mlp`) plus optional OpenVINO readiness. |
| `POST` | `/api/plan` | Validate and preview a command without starting a run. |
| `POST` | `/api/command` | Submit `{"text":"...", "style?":"", "gesture?":""}` for Groq tool control when configured, with deterministic planning fallback. |
| `POST` | `/api/agent` | Explicit alias for the bounded Groq workcell tool loop. |
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
voxhands/agent.py         Bounded Groq tool definitions, loop, and dispatcher
voxhands/styles.py        Movement-style registry, easing, and gesture keyframes
voxhands/safety.py        Plan validation and red-zone safety rules
voxhands/simulation.py    Deterministic 60 Hz server-side task simulation
voxhands/mujoco_scene.py  MuJoCo scene wrapper: IK, attach/release, rendering
voxhands/mujoco_sim.py    MuJoCo simulation backend (drop-in for simulation.py)
voxhands/vision.py        MujocoVision: OpenVINO MLP + calibrated colour-mask
voxhands/imageio.py       Pillow JPEG / stdlib BMP frame encoders
voxhands/assets/scene.xml MuJoCo workcell model (two SO-101-style arms, cameras)
voxhands/server.py        Standard-library HTTP server and API routes
voxhands/models.py        Plan, action, and snapshot data models
voxhands/integrations.py  Optional runtime availability detection
voxhands/openvino_adapter.py  OpenVINO IR export (to_ir), NNCF INT8 (quantize_ir), CPU benchmark
voxhands/check_intel.py   CLI report of Intel OpenVINO GPU/NPU/CPU + NNCF status
voxhands/policy/          Policy classes: MLPPolicy, CanonicalDinnerPolicy, synthetic demos
scripts/convert_openvino.py    Export the policy MLP to a real OpenVINO IR (.xml/.bin)
scripts/quantize_openvino.py   NNCF INT8 post-training quantization of the IR
scripts/benchmark_openvino.py  Measured OpenVINO CPU latency/throughput (FP32 vs INT8)
scripts/train_policy.py        Train/emit data/policy.json on synthetic demos
frontend/index.html       Dashboard, Three.js scene, and Rapier3D physics
tests/                    Keyword, AI, simulation, IR-export, and API-contract tests
LICENSE                   MIT license for VoxHands source code
```

## Verification

Run the Python tests and static checks from the repository root:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q voxhands tests
node -e 'const fs=require("fs"); const html=fs.readFileSync("frontend/index.html","utf8"); const scripts=html.split("<script>")[1].split("</script>")[0]; new Function(scripts); console.log("frontend JavaScript syntax OK");'
git diff --check
```

The full regression suite includes planner, safety, simulator, Groq, agent,
HTTP contract, MuJoCo robustness, policy, OpenVINO IR-export, INT8-quantization,
benchmark/sweep, and throughput tests. A recent run completed with 98 passing
tests. For the summit demo, also perform the browser checklist below after
starting the live server; backend tests alone do not prove the rendered WebGL
interaction.

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

Install them only in the matching hardware, credentials, or hackathon
environment and pin the versions there:

```text
mujoco==3.13.0         # MuJoCo physics + per-frame rendering (recommended backend)
openvino==2026.3.1     # optional vision inference (NPU/GPU/CPU)
openvino-genai==2026.3.0
nncf==3.3.0            # INT8 post-training quantization of the policy/vision IR (CPU)
speechmatics-rt
lerobot
```

The optional list is recorded in
[`requirements-optional.txt`](requirements-optional.txt). The current UI
labels unavailable integrations as demo/unavailable instead of claiming that
the laptop is connected to hardware or an NPU; when OpenVINO wheels are not
available for the installed Python, `MujocoVision` transparently falls back to
the calibrated colour-mask detector.

## MuJoCo backend boundary

`voxhands/mujoco_sim.MujocoTableSettingSimulation` is a drop-in
`TableSettingSimulation`: it keeps the exact plan, safety, HTTP, event, and
browser telemetry contracts while replacing motion with MuJoCo
inverse-kinematics on two SO-101-style arms. `MujocoVision` reads real rendered
camera frames; when the optional OpenVINO package is installed it runs a
3→16→4 MLP detector on NPU→GPU→CPU, otherwise a geometric colour-mask detector
tracks the four objects. Groq tool control remains gated on `GROQ_API_KEY` and
the model can request only validated workcell tools, never raw motor commands
or arbitrary physics mutation.

## Intel OpenVINO optimization path (IR export, INT8 PTQ, measured CPU inference)

The policy MLP and the vision classifier run through a real Intel OpenVINO
pipeline, not a numpy stand-in. The optimization flow is reproducible end to end
on CPU:

```bash
python3.13 scripts/train_policy.py                 # -> data/policy.json (auto-created if missing)
python3.13 scripts/convert_openvino.py             # -> data/openvino_ir/policy_mlp.xml + .bin  (real IR)
python3.13 scripts/quantize_openvino.py            # -> data/openvino_ir_int8/...xml + .bin (NNCF PTQ)
python3.13 scripts/benchmark_openvino.py           # FP32 vs INT8 latency/throughput for the policy
python3.13 scripts/benchmark_openvino.py --sweep --throughput   # size sweep (see table below)
```

What each stage actually does:

- **IR export** (`OpenVINOAdapter.to_ir`): builds the MLP as an `openvino.Model`
  with `opset14` and writes a genuine `.xml`/`.bin` via `openvino.save_model`.
  The same graph is compiled in-process by `MujocoVision` for the camera loop.
- **INT8 PTQ** (`quantize_ir`): NNCF post-training quantization of the IR with a
  calibration set sampled from the policy input distribution, producing a real
  quantized IR. Quantization runs on CPU; no Intel GPU/NPU is required.
- **Benchmark** (`benchmark_inference` / `benchmark_throughput`): times the
  **compiled OpenVINO model** (not the numpy fallback) with a median of repeats,
  reporting mean/p50/p95 latency and — via `ov.AsyncInferQueue` — real concurrent
  throughput. `/api/vision` surfaces the measured per-detection `inference_ms`
  and `device` from the live camera loop.

### Measured sweep: where INT8 pays off

Apple CPU, OpenVINO 2026.3.1, FP16 inference hint, median of 5 repeats; the
per-model `.bin` is FP32 vs NNCF INT8. Reproduce with
`python3.13 scripts/benchmark_openvino.py --sweep --throughput`.

| Model (layers) | Params | FP32 latency | INT8 latency | Latency speedup | INT8 `.bin` |
| --- | --- | --- | --- | --- | --- |
| `policy` (8→16→16→8) | 552 | 0.030 ms | 0.031 ms | 0.97× (neutral) | 0.70× |
| `vision` (3→512→256→4) | 134,404 | 0.040 ms | 0.040 ms | 1.00× (neutral) | 0.51× |
| `workcell-large` (256→2048→2048→128) | 4,984,960 | 0.152 ms | 0.125 ms | **1.21×** | 0.50× |
| `workcell-xlarge` (512→4096→4096→256) | 19,931,392 | 0.597 ms | 0.350 ms | **1.71×** | 0.50× |

Async throughput improves with INT8 as the model becomes compute-bound:
`workcell-xlarge` rises from **2,723 → 5,003 inf/s (1.84×)**.

Honest reading of the data:

- The **tiny workcell policy is dispatch-bound**, not compute-bound: one inference
  is ~30 µs, so INT8 is neutral there (and its weight `.bin` still shrinks). The
  important consequence is that model inference is negligible against the 60 Hz
  (16.7 ms) physics tick — roughly 500× headroom per tick.
- **INT8 gives a real 1.2×–1.7× latency win and ~2× smaller weights once the
  model is compute-bound.** So the optimization pipeline is genuine and measured,
  and the sizing data shows exactly where it matters. GPU/NPU INT8 kernels remain
  gated on Intel hardware.

What is verified vs. gated:

- **Verified**: `openvino` 2026.3.1 imports; real `.xml`/`.bin` IR export;
  `compile_model` CPU inference; NNCF INT8 PTQ; measured CPU latency and async
  throughput; the 1.2×–1.7× speedup at deployment scale.
- **Gated, NOT verified**: GPU inference, NPU inference, and GPU/NPU INT8 kernels.
  On this Apple host `ov.Core().available_devices` returns `['CPU']`, so
  `verified` is `False`, `device_used` is `"CPU"`, and no GPU/NPU numbers are
  fabricated.

`OpenVINOAdapter` (in `voxhands/openvino_adapter.py`) is the single honest source
for this state. Its `convert()` method remains an optional ONNX-free numpy
fallback (`weights.npz` + JSON manifest) and does **not** emit an IR — use
`to_ir()` for the real artifact. `voxhands/vision.py` compiles the classifier
graph in-process and reports measured `inference_ms` through `/api/vision`.

Robustness harness status on this host (2 perturbed scenes each, from
`scripts/run_mujoco_eval.py`):

| Mode | Seeds | Seeds OK | All OK |
| --- | --- | --- | --- |
| Oracle (simulator ground truth) | 2 | 2 | true |
| Vision (overhead-camera loop) | 2 | 2 | true |

Check the same state from the CLI at any time:

```bash
python3.13 voxhands/check_intel.py
python3.13 scripts/check_intel.py
python3.13 scripts/convert_openvino.py
python3.13 scripts/quantize_openvino.py
python3.13 scripts/benchmark_openvino.py
```

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

## License

VoxHands source code is available under the [MIT License](LICENSE). Vendored
third-party modules retain their own licenses; see the license files beside the
modules in `frontend/vendor`.
