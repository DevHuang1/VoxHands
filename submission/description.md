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

> A voice-first dual-arm tabletop assistant where language models only propose
> intent. A deterministic Python simulator validates and executes every action,
> with real Intel OpenVINO IR export, NNCF INT8 quantization, and measured CPU
> inference.

## Technology and category tags

`Physical AI` · `Robotics` · `OpenVINO` · `Edge AI` · `AI Agents` · `Simulation` · `Python`

## Long description

VoxHands is a voice-first dual-arm tabletop assistant that separates language
from action. A Groq-powered tool loop turns a spoken or typed command such as
"set the table for two" into a bounded set of workcell tools — never raw motor
commands. Before anything moves, a deterministic Python validator enforces
workcell bounds, collision rules and no-go zones; unsupported or unsafe requests
are refused outright rather than silently reinterpreted. The Python simulation
owns the authoritative state of every object and arm, while the browser renders
the 3D workcell with Three.js and confirms contact through Rapier3D physics.

The project also carries a real Intel optimization path rather than a stand-in:
the policy network is exported to a genuine OpenVINO IR (`openvino.save_model`),
quantized to INT8 with NNCF post-training quantization, and benchmarked on CPU.
Measured results show INT8 reaching 1.21×–1.71× speedup on compute-bound models
and 5,003 inferences/second asynchronous throughput, while we state plainly that
the tiny workcell policy is dispatch-bound at ~30 µs and gains nothing from INT8.

That honesty is the point. VoxHands reports what is verified and gates what is
not: GPU and NPU acceleration stay explicitly unverified until they run on Intel
Core Ultra hardware. The result is a reproducible testbed for physical AI —
useful to robotics integrators, physical-AI researchers, and silicon vendors who
need a safe command layer and an honest optimization baseline. The simulator is
MIT-licensed and runs on the Python standard library; the optional MuJoCo
backend, OpenVINO vision adapter and browser physics layer activate when present
and degrade gracefully when they are not.

## Honest limitations (state these plainly)

- Acceleration numbers are CPU-only. The available host exposes `CPU` alone, so
  GPU/NPU results are gated, not estimated.
- Grasping and placement are simplified; flowing liquid is not simulated.
- The deployed free-tier instance sleeps when idle and can take about a minute
  to wake; the first request may be slow.
- No physical robot is driven. Everything executes in simulation.
