# VoxHands — submission copy

Paste-ready text for the lablab.ai project page. Character counts are included so
the field limits can be checked at a glance.

- **Project title:** VoxHands — Safe Physical AI in a Simulated Workcell
- **Live demo:** https://voxhands.onrender.com/
- **Repository:** https://github.com/DevHuang1/VoxHands
- **Video:** `VoxHands_demo_1080p.mp4` (115 s, 1920×1080)
- **Slides:** `VoxHands-deck.pdf` (10 pages, 16:9)
- **Cover:** `cover.png` (1920×1080, 16:9)
- **Team:** Thu Ta Sitt — Lead Developer; Yan Myo Thuyako — Software Engineer

## Short description (255 character limit)

> Voice-first dual-arm tabletop assistant: LLMs propose bounded workcell tools,
> a Python simulator executes them, and a live MuJoCo backend runs two
> SO-101-style arms with OpenVINO inference on CPU only — no Core Ultra hardware
> used, so no GPU/NPU claim.

## Technology and category tags

`Physical AI` · `Robotics` · `OpenVINO` · `Edge AI` · `AI Agents` · `Simulation` · `Python`

## Long description (2000 character limit — 1,994 used)

VoxHands is a voice-first dual-arm tabletop assistant that separates language from action. A Groq-powered tool loop turns a command such as "set the table for two" into a bounded set of workcell tools, never raw motor commands. A deterministic Python validator enforces workcell bounds, collision rules, and no-go safety zones before anything moves, refusing unsafe requests outright. The Python simulation owns authoritative object and arm state, while the browser renders the workcell and confirms contact through Rapier3D physics at 60 Hz.

The deployed service runs a real MuJoCo backend: two SO-101-style six-DOF arms driven by Jacobian inverse kinematics, kinematic grasp and release, and gravity-settled placement, with overhead and operator camera views. MuJoCo's offscreen renderer needs an EGL/OSMesa GL stack that Render's native Python runtime cannot install, so the blueprint builds a Docker image (Python 3.13, mujoco, Pillow, OpenVINO); the robot adapter now reports available live instead of falling back to the deterministic simulator.

VoxHands also carries a real Intel optimization path, not a stand-in: the policy network is exported to a genuine OpenVINO IR, quantized to INT8 with NNCF post-training quantization, and benchmarked on CPU. INT8 reaches 1.21x to 1.71x speedup on compute-bound models and 5,003 inferences/second asynchronous throughput, while we state plainly that the tiny workcell policy is dispatch-bound and gains nothing from INT8. All figures are measured on the live host.

That honesty is the point: we report what is verified and gate what is not, with GPU and NPU acceleration explicitly unverified until it runs on Intel Core Ultra hardware. The result is a reproducible, MIT-licensed testbed for physical AI where optional backends, MuJoCo, OpenVINO vision, and browser physics, activate when present and degrade gracefully when absent. Grasping and placement remain simplified and no physical robot is driven: everything executes in simulation.

## Additional info (2000 character limit — 1,996 used)

How it works. A browser dashboard polls one stateful Python server over HTTP. A spoken or typed command goes to Groq, which runs a bounded tool loop: the model may only request validated workcell tools such as plan, pick, place, reset, never raw motor commands or arbitrary physics mutation. The returned plan is checked by a deterministic safety validator against workcell bounds, arm collision rules, and a no-go zone before motion starts; unsafe requests are refused outright. The simulation then steps at a fixed 60 Hz, generating timestamped minimum-jerk trajectories and serving authoritative state, mirrored in the browser with Three.js and confirmed through Rapier3D contact and settle telemetry.

The optional MuJoCo backend replaces that motion with physics: two SO-101-style six-DOF arms are driven by Jacobian inverse kinematics, objects attach kinematically at the grip site, and placement is handed to gravity so they physically settle. Overhead and operator cameras render offscreen through EGL; the live deployment builds a Docker image so the GL stack is present. A vision layer reads the rendered frame, detecting the four objects via calibrated colour mask and upgrading to OpenVINO inference when present.

The Intel path is genuine, not a stand-in: the policy MLP is exported to a real OpenVINO IR with openvino.save_model, quantized to INT8 with NNCF post-training quantization, and benchmarked on CPU. Scripts train the policy, convert the IR, quantize it, and sweep FP32 versus INT8 latency and throughput, all published from the live host at /api/intel. Plainly: the available host exposes CPU only, no Intel Core Ultra hardware was available, so GPU and NPU acceleration remain unverified rather than estimated and the brand string is reported verbatim. The core runs on the Python standard library with zero required dependencies, optional backends activate when installed and degrade gracefully when absent, and the project is MIT-licensed and reproducible end to end.

## Honest limitations (state these plainly)

- Acceleration numbers are CPU-only. The available host exposes `CPU` alone, so
  GPU/NPU results are gated, not estimated.
- Grasping and placement are simplified; flowing liquid is not simulated.
- The deployed free-tier instance sleeps when idle and can take about a minute
  to wake; the first request may be slow.
- No physical robot is driven. Everything executes in simulation.
