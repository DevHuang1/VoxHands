# VoxHands

VoxHands is a voice-first dual-arm tabletop assistant for the AI Infra Summit hackathon concept.

The first MVP is intentionally dependency-light: it runs a deterministic tabletop simulation and a live browser dashboard with no install step beyond Python 3.10+. The control boundaries are already separated so MuJoCo, OpenVINO, and Speechmatics can be plugged in without rewriting the product flow.

## Run it

```bash
cd /Users/yuza/Desktop/VoxHands
python3 run.py
```

Open <http://127.0.0.1:8000> and try:

```text
Set the table for two. Put the blue plate on the left and the cup on the right. Avoid the red zone.
```

The dashboard shows the command transcript, validated action plan, simulated camera scene, dual-arm progress, and event timeline.

## Current MVP boundaries

- Implemented: command parsing, structured plans, safety validation, parallel left/right arm execution, recovery-ready event logging, HTTP API, and dashboard.
- Demo mode: the tabletop world is a deterministic server-side simulator so it runs on any laptop.
- Adapter boundaries: runtime status detection for OpenVINO and Speechmatics is included; real credentials and hardware are not assumed.
- Next integration: replace `TableSettingSimulation` with the Intel-provided MuJoCo/LeRobot backend while preserving the plan and safety contracts.

## Architecture

```text
Browser command
      |
      v
POST /api/command -> Planner -> Safety validator -> Dual-arm simulator
                                                |
                                                v
                                      GET /api/state -> Dashboard
```

The planner produces JSON-like actions. The simulator, not a language model, owns execution. This makes it safe to swap in a local OpenVINO planner later without allowing free-form model output to drive motors directly.

## Optional partner dependencies

The MVP does not require these packages. Add them only when the matching hardware or credentials are available, and pin versions to the hackathon-provided environment:

```text
openvino==2026.3.0
openvino-genai==2026.3.0
speechmatics-rt
mujoco
lerobot
```

The current dashboard truthfully labels unavailable integrations instead of pretending the laptop is running on an NPU or connected to Speechmatics.

