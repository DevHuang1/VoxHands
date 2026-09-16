import React from "react";
import {AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame} from "remotion";
import {Slide} from "./components/Slide";
import {WorkcellPlane} from "./components/WorkcellPlane";
import {IconBadge, IconName} from "./components/Icon";
import {BenchmarkTable} from "./components/BenchmarkTable";
import {StackChips, Stats} from "./components/Stats";
import {Terminal} from "./components/Terminal";
import {Accent, theme} from "./theme";

export const SLIDE_DURATION = 150;

type Scene = {
  eyebrow: string;
  title: string;
  subtitle?: string;
  caption: string;
  accent?: Accent;
  visual: React.ReactNode;
};

const icon = (name: IconName, color: string = theme.cyan) => <IconBadge name={name} color={color} />;

const sweepRows = [
  {model: "policy", params: "552", fp32: "0.030 ms", int8: "0.031 ms", speedup: 0.97, bin: "0.70x"},
  {model: "vision", params: "134k", fp32: "0.040 ms", int8: "0.040 ms", speedup: 1.0, bin: "0.51x"},
  {model: "workcell-large", params: "5.0M", fp32: "0.152 ms", int8: "0.125 ms", speedup: 1.21, bin: "0.50x", highlight: true},
  {model: "workcell-xlarge", params: "19.9M", fp32: "0.597 ms", int8: "0.350 ms", speedup: 1.71, bin: "0.50x", highlight: true},
];

export const scenes: Scene[] = [
  {
    eyebrow: "Voice-first tabletop assistant",
    title: "VoxHands",
    subtitle: "From spoken intent to coordinated motion.",
    caption: "What if setting a table started with a sentence?",
    visual: <WorkcellPlane highlight="none" />,
  },
  {
    eyebrow: "The hard part",
    title: "Language models do not do physics.",
    subtitle: "VoxHands splits intent from execution.",
    caption: "Let the model decide. Let the simulator own reality.",
    visual: icon("boundary"),
  },
  {
    eyebrow: "Four objects · two arms",
    title: "The shared workcell.",
    subtitle: "A plate, cup, fork, and spoon share one workspace.",
    caption: "Two arms share a workcell with a plate, cup, fork, and spoon.",
    visual: <WorkcellPlane highlight="arms" />,
  },
  {
    eyebrow: "Command input",
    title: "Voice or text.",
    subtitle: "Type a task or use the browser microphone when available.",
    caption: "Type a command, or use your browser's microphone when available.",
    visual: icon("voice"),
  },
  {
    eyebrow: "Language layer",
    title: "Groq turns a sentence into a tool call.",
    subtitle: "A bounded local tool loop, not free-form code.",
    caption: "The model proposes only validated workcell tools.",
    visual: icon("ai"),
  },
  {
    eyebrow: "Tool boundary",
    title: "Ten tools. No raw motors.",
    subtitle: "Place, move, grip, home, wait, pause, resume, stop, reset, observe.",
    caption: "No raw motor commands and no arbitrary physics mutation.",
    visual: icon("boundary"),
  },
  {
    eyebrow: "Safety gate",
    title: "Python checks the plan.",
    subtitle: "Every request is validated before motion begins.",
    caption: "Python checks each request before the simulator starts moving.",
    accent: "green",
    visual: icon("shield", theme.green),
  },
  {
    eyebrow: "Authoritative state",
    title: "The server owns truth.",
    subtitle: "The browser draws and interpolates; it never decides.",
    caption: "Server state is authoritative; the browser renders telemetry.",
    visual: icon("layers"),
  },
  {
    eyebrow: "Feedback loop",
    title: "Tools report back.",
    subtitle: "Tool results help explain what happened after each action.",
    caption: "It receives tool results to help explain what happened.",
    visual: icon("refresh"),
  },
  {
    eyebrow: "Motion contract",
    title: "60 Hz, minimum-jerk.",
    subtitle: "Timestamped trajectories with grasp alignment under six millimetres.",
    caption: "Motion is sampled at 60 Hz and validated by physics.",
    visual: icon("robot"),
  },
  {
    eyebrow: "Browser physics",
    title: "Rapier3D confirms contact.",
    subtitle: "Three.js renders; Rapier3D checks grasp, release, and settling.",
    caption: "The browser confirms contact while the server owns the plan.",
    visual: icon("diamond", theme.blue),
  },
  {
    eyebrow: "Container logic",
    title: "Spoon into cup.",
    subtitle: "Placement rules distinguish inside from underneath.",
    caption: "Placement rules distinguish an object inside a cup from one underneath it.",
    visual: icon("layers"),
  },
  {
    eyebrow: "Relationships",
    title: "The cup carries its spoon.",
    subtitle: "Move the cup onto the plate while the spoon stays inside.",
    caption: "Next, move the cup onto the plate, with the spoon still inside.",
    visual: <WorkcellPlane highlight="cup" />,
  },
  {
    eyebrow: "Safety refusal",
    title: "Blocked, not guessed.",
    subtitle: "Red-zone requests are refused, never reinterpreted.",
    caption: "Red-zone requests stop before motion and never silently change the task.",
    accent: "red",
    visual: <WorkcellPlane highlight="barrier" />,
  },
  {
    eyebrow: "The software stack",
    title: "Python, Three.js, Rapier, Groq.",
    subtitle: "Python owns task state; the browser draws and simulates.",
    caption: "Python owns task state. Groq adds a cloud language and tool layer.",
    visual: <StackChips chips={["Python", "Three.js", "Rapier3D", "Groq", "MuJoCo", "OpenVINO"]} />,
  },
  {
    eyebrow: "Intel track · OpenVINO",
    title: "A real OpenVINO IR.",
    subtitle: "The policy MLP is exported to .xml and .bin with openvino.save_model.",
    caption: "Not a numpy stand-in: a genuine OpenVINO Runtime artifact.",
    visual: icon("chip"),
  },
  {
    eyebrow: "IR export",
    title: "numpy weights become an IR.",
    subtitle: "opset14 graph to openvino.Model, saved as .xml plus .bin.",
    caption: "The same graph the vision loop compiles on CPU.",
    visual: (
      <Terminal
        lines={[
          {prefix: "$", text: "python3.13 scripts/convert_openvino.py"},
          {text: "Real OpenVINO IR written: data/openvino_ir/policy_mlp.xml", color: theme.green},
          {text: "policy_mlp.xml (8857 B) + policy_mlp.bin (1104 B)", color: theme.muted},
          {text: "IR input dim=8, output dim=8", color: theme.muted},
        ]}
      />
    ),
  },
  {
    eyebrow: "Int8 post-training quantization",
    title: "INT8 in one command.",
    subtitle: "NNCF quantizes the IR on CPU — no Intel GPU required.",
    caption: "INT8 post-training quantization halves the model weights.",
    visual: (
      <Terminal
        lines={[
          {prefix: "$", text: "python3.13 scripts/quantize_openvino.py"},
          {text: "INT8 IR written: data/openvino_ir_int8/policy_mlp_int8.xml", color: theme.green},
          {text: "FP32 vs INT8 max abs diff=0.047363 (rel=5.67%)", color: theme.muted},
          {text: "device=CPU · calibration=256 samples", color: theme.muted},
        ]}
      />
    ),
  },
  {
    eyebrow: "Measured on CPU",
    title: "1.7x faster where it matters.",
    subtitle: "At deployment scale, INT8 wins on latency and async throughput.",
    caption: "19.9M params: 0.597 ms to 0.350 ms; 2,723 to 5,003 inferences per second.",
    visual: <BenchmarkTable rows={sweepRows} />,
  },
  {
    eyebrow: "Right-sized",
    title: "The workcell policy is tiny.",
    subtitle: "One inference is ~30 µs, dispatch-bound, so INT8 is neutral there.",
    caption: "Model inference is roughly 500x smaller than the 16.7 ms physics tick.",
    visual: (
      <Stats
        items={[
          {value: "30 µs", label: "per policy inference", color: theme.cyan},
          {value: "16.7 ms", label: "per 60 Hz physics tick", color: theme.blue},
        ]}
      />
    ),
  },
  {
    eyebrow: "Honest limits",
    title: "Verified versus gated.",
    subtitle: "CPU IR, INT8, and benchmarks are verified. GPU and NPU stay gated.",
    caption: "No fabricated GPU or NPU numbers.",
    accent: "amber",
    visual: icon("gauge", theme.amber),
  },
  {
    eyebrow: "Current limits",
    title: "A clear prototype.",
    subtitle: "Grasps and placement are simplified; flowing coffee is not simulated.",
    caption: "It does not yet control physical robots or simulate flowing coffee.",
    accent: "amber",
    visual: icon("diamond", theme.amber),
  },
  {
    eyebrow: "VoxHands",
    title: "Build on the infrastructure, not the demo.",
    subtitle: "numpy to OpenVINO IR to INT8 to measured CPU inference.",
    caption: "VoxHands — voice-first dual-arm control with a real optimization pipeline.",
    visual: <WorkcellPlane highlight="plate" />,
  },
];

export const VOXHANDS_DURATION = scenes.length * SLIDE_DURATION;

export const VoxHands: React.FC = () => {
  const frame = useCurrentFrame();
  const total = VOXHANDS_DURATION;

  return (
    <AbsoluteFill style={{background: theme.bg}}>
      <Audio
        src={staticFile("soundtrack.mp3")}
        volume={(f) =>
          interpolate(f, [total - 45, total - 5], [1, 0], {
            extrapolateLeft: "clamp",
            extrapolateRight: "clamp",
          })
        }
      />
      {scenes.map((scene, i) => (
        <Sequence
          key={scene.title}
          from={i * SLIDE_DURATION}
          durationInFrames={SLIDE_DURATION}
          name={scene.eyebrow}
        >
          <Slide
            eyebrow={scene.eyebrow}
            title={scene.title}
            subtitle={scene.subtitle}
            caption={scene.caption}
            accent={scene.accent}
            index={i + 1}
            total={scenes.length}
            durationInFrames={SLIDE_DURATION}
            overallProgress={(frame + 1) / total}
          >
            {scene.visual}
          </Slide>
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
