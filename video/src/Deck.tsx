import React from "react";
import {AbsoluteFill, Img, staticFile, useCurrentFrame} from "remotion";
import {Accent, accentColor, theme} from "./theme";
import {BenchmarkTable, Row} from "./components/BenchmarkTable";
import {Icon, IconName} from "./components/Icon";
import {StackChips} from "./components/Stats";
import {Terminal} from "./components/Terminal";
import {WorkcellPlane} from "./components/WorkcellPlane";

export const DECK_URL = "voxhands.onrender.com";
const TOTAL = 10;

/* ------------------------------------------------------------------ layout */

const DeckFrame: React.FC<{
  index: number;
  eyebrow: string;
  title: string;
  subtitle?: string;
  accent?: Accent;
  children: React.ReactNode;
  dense?: boolean;
}> = ({index, eyebrow, title, subtitle, accent = "cyan", children, dense = false}) => {
  const color = accentColor(accent);
  return (
    <AbsoluteFill style={{fontFamily: theme.font, color: theme.text}}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(circle at 84% -10%, ${color}22 0%, transparent 36%), linear-gradient(180deg, #06101a 0%, ${theme.bg} 62%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 0,
          height: 8,
          background: `linear-gradient(90deg, ${color}, transparent 72%)`,
          opacity: 0.55,
        }}
      />

      <div style={{position: "absolute", left: 110, top: 84, right: 110}}>
        <div
          style={{
            fontSize: 19,
            fontWeight: 700,
            letterSpacing: "0.32em",
            textTransform: "uppercase",
            color,
            opacity: 0.9,
          }}
        >
          {eyebrow}
        </div>
        <h1
          style={{
            margin: "22px 0 0",
            fontSize: dense ? 66 : 76,
            lineHeight: 1.02,
            letterSpacing: "-0.04em",
            fontWeight: 800,
          }}
        >
          {title}
        </h1>
        {subtitle ? (
          <p style={{margin: "18px 0 0", fontSize: 28, lineHeight: 1.4, color: theme.muted, maxWidth: 1360}}>
            {subtitle}
          </p>
        ) : null}
      </div>

      <div style={{position: "absolute", left: 110, right: 110, top: subtitle ? 330 : 300, bottom: 128}}>
        {children}
      </div>

      <div
        style={{
          position: "absolute",
          left: 110,
          right: 110,
          bottom: 56,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          fontFamily: theme.mono,
          fontSize: 18,
          color: theme.faint,
          letterSpacing: "0.08em",
        }}
      >
        <span>VOXHANDS · SAFE PHYSICAL AI IN A SIMULATED WORKCELL</span>
        <span>
          {String(index).padStart(2, "0")} / {TOTAL}
        </span>
      </div>
    </AbsoluteFill>
  );
};

const Card: React.FC<{
  title: string;
  body: string;
  accent?: string;
  icon?: IconName;
  tag?: string;
  compact?: boolean;
}> = ({title, body, accent = theme.cyan, icon, tag, compact = false}) => (
  <div
    style={{
      flex: 1,
      border: `1px solid ${theme.line}`,
      borderRadius: 18,
      background: "rgba(8,19,29,0.86)",
      padding: compact ? "22px 26px 24px" : "30px 30px 32px",
      display: "flex",
      flexDirection: "column",
      gap: compact ? 12 : 16,
    }}
  >
    {icon ? <Icon name={icon} color={accent} size={64} /> : null}
    {tag ? (
      <span
        style={{
          fontFamily: theme.mono,
          fontSize: 15,
          letterSpacing: "0.14em",
          color: accent,
          textTransform: "uppercase",
        }}
      >
        {tag}
      </span>
    ) : null}
    <div style={{fontSize: compact ? 26 : 30, fontWeight: 700, letterSpacing: "-0.02em"}}>{title}</div>
    <div style={{fontSize: compact ? 19 : 21, lineHeight: 1.5, color: theme.muted}}>{body}</div>
  </div>
);

const Row_: React.FC<{n: number; title: string; body: string; accent?: string}> = ({
  n,
  title,
  body,
  accent = theme.cyan,
}) => (
  <div style={{display: "flex", gap: 26, alignItems: "flex-start"}}>
    <div
      style={{
        minWidth: 58,
        height: 58,
        borderRadius: 14,
        border: `1px solid ${accent}55`,
        background: `${accent}14`,
        color: accent,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 26,
        fontWeight: 800,
        fontFamily: theme.mono,
      }}
    >
      {n}
    </div>
    <div>
      <div style={{fontSize: 29, fontWeight: 700, letterSpacing: "-0.015em"}}>{title}</div>
      <div style={{marginTop: 8, fontSize: 21, lineHeight: 1.5, color: theme.muted, maxWidth: 820}}>{body}</div>
    </div>
  </div>
);

/* ------------------------------------------------------------------- slides */

const SlideTitle: React.FC = () => (
  <DeckFrame
    index={1}
    eyebrow="AI Infra Summit Hackathon · Physical AI"
    title="VoxHands"
    subtitle="Safe physical AI in a simulated workcell — language interprets intent, deterministic systems own execution."
    accent="cyan"
  >
    <div style={{display: "flex", alignItems: "center", gap: 60, height: "100%"}}>
      <div style={{flex: 1}}>
        <div style={{fontSize: 30, lineHeight: 1.5, color: theme.text, maxWidth: 720}}>
          A voice-first dual-arm tabletop assistant with a validated tool loop, an authoritative Python
          simulation, and a real Intel OpenVINO optimization path.
        </div>
        <div
          style={{
            marginTop: 46,
            display: "flex",
            flexDirection: "column",
            gap: 14,
            fontFamily: theme.mono,
            fontSize: 22,
          }}
        >
          <span style={{color: theme.cyan}}>▸ {DECK_URL}</span>
          <span style={{color: theme.muted}}>Thu Ta Sitt — Lead Developer</span>
          <span style={{color: theme.muted}}>Yan Myo Thuyako — Software Engineer</span>
        </div>
      </div>
      <div style={{width: 720, display: "flex", justifyContent: "center"}}>
        <WorkcellPlane highlight="arms" />
      </div>
    </div>
  </DeckFrame>
);

const SlideProblem: React.FC = () => (
  <DeckFrame
    index={2}
    eyebrow="The gap between language and action"
    title="Language models do not do physics."
    subtitle="Four failure modes appear the moment generated text is allowed to drive a machine."
    accent="red"
  >
    <div style={{display: "flex", flexDirection: "column", gap: 22, height: "100%"}}>
      <div style={{display: "flex", gap: 22}}>
        <Card
          icon="ai"
          accent={theme.red}
          tag="Unconstrained intent"
          title="Fluent, but impossible"
          body="A model can generate grammatically correct instructions that are physically impossible or unsafe."
        />
        <Card
          icon="shield"
          accent={theme.red}
          tag="No verification"
          title="Determinism is missing"
          body="Direct execution of AI output skips the deterministic validation robotics and physical safety require."
        />
        <Card
          icon="refresh"
          accent={theme.red}
          tag="State drift"
          title="Imagined state diverges"
          body="Without an authoritative simulator, the gap between the 'imagined' and real workcell grows every step."
        />
      </div>
      <div
        style={{
          border: `1px solid ${theme.red}44`,
          borderRadius: 16,
          background: "rgba(30,10,12,0.55)",
          padding: "24px 30px",
          fontFamily: theme.mono,
          fontSize: 22,
          color: theme.text,
          display: "flex",
          justifyContent: "space-between",
          gap: 24,
        }}
      >
        <span>
          <span style={{color: theme.cyan}}>$ </span>
          Move the plate to the ground
        </span>
        <span style={{color: theme.red, fontWeight: 700}}>UNSAFE_ACTION_DETECTED · rejected before motion</span>
      </div>
    </div>
  </DeckFrame>
);

const SlideSolution: React.FC = () => (
  <DeckFrame
    index={3}
    eyebrow="The solution"
    title="A bounded tool loop."
    subtitle="The model proposes intent. A deterministic layer decides what may become motion."
    accent="green"
  >
    <div style={{display: "flex", flexDirection: "column", gap: 30, paddingTop: 12}}>
      <Row_
        n={1}
        title="Intent interpretation — Groq"
        body="The language model maps a conversational command onto a strictly defined set of workcell tools. It cannot emit raw motor commands or arbitrary physics mutations."
        accent={theme.cyan}
      />
      <Row_
        n={2}
        title="Safety validation — deterministic Python"
        body="A validator enforces workcell bounds, collision rules and no-go zones before any motion begins. Red-zone requests are refused, never silently reinterpreted."
        accent={theme.green}
      />
      <Row_
        n={3}
        title="Authoritative execution — Python simulator"
        body="Python owns the true state of every object and arm; the browser renders 60 Hz motion and reports contact telemetry for confirmation only."
        accent={theme.blue}
      />
    </div>
  </DeckFrame>
);

const SlidePipeline: React.FC = () => (
  <DeckFrame
    index={4}
    eyebrow="From command to motion"
    title="One validated path, four stages."
    subtitle="Every request travels the same pipeline — no shortcut from text to torque."
    accent="blue"
  >
    <div style={{display: "flex", alignItems: "center", gap: 18, paddingTop: 40}}>
      {[
        {tag: "Input", title: "Natural language", body: "Typed command or browser microphone."},
        {tag: "Plan", title: "Tools + intent", body: "Groq selects a validated tool call."},
        {tag: "Simulate", title: "60 Hz motion", body: "Python engine owns the authoritative state."},
        {tag: "Telemetry", title: "Contact + settling", body: "Rapier3D reports physics back to the server."},
      ].map((step, i) => (
        <React.Fragment key={step.tag}>
          <div
            style={{
              flex: 1,
              border: `1px solid ${theme.line}`,
              borderRadius: 16,
              background: "rgba(8,19,29,0.86)",
              padding: "28px 26px",
              minHeight: 240,
            }}
          >
            <div
              style={{
                fontFamily: theme.mono,
                fontSize: 15,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: theme.blue,
              }}
            >
              {step.tag}
            </div>
            <div style={{marginTop: 16, fontSize: 27, fontWeight: 700}}>{step.title}</div>
            <div style={{marginTop: 12, fontSize: 20, lineHeight: 1.5, color: theme.muted}}>{step.body}</div>
          </div>
          {i < 3 ? <div style={{color: theme.faint, fontSize: 34}}>›</div> : null}
        </React.Fragment>
      ))}
    </div>
  </DeckFrame>
);

const SlideSafety: React.FC = () => (
  <DeckFrame
    index={5}
    eyebrow="Safety is the product"
    title="Refusal is a feature."
    subtitle="In physical AI, the ability to say no is more valuable than the ability to act."
    accent="green"
  >
    <div style={{display: "flex", gap: 22, height: "100%"}}>
      <Card
        icon="boundary"
        accent={theme.green}
        tag="Workcell bounds"
        title="Strict XYZ limits"
        body="Arms cannot move outside the tabletop volume, regardless of what the model requests."
      />
      <Card
        icon="shield"
        accent={theme.green}
        tag="No-go zones"
        title="Authoritative barriers"
        body="Red-zone intersections block motion and placement before a trajectory is ever executed."
      />
      <Card
        icon="layers"
        accent={theme.green}
        tag="Authoritative state"
        title="The server owns truth"
        body="Browser physics is telemetry. Object poses are decided server-side, so state cannot drift."
      />
      <Card
        icon="bolt"
        accent={theme.green}
        tag="Protective stop"
        title="Telemetry can halt"
        body="A collision reported by browser physics triggers an immediate server-side stop of all plans."
      />
    </div>
  </DeckFrame>
);

const SlideDemo: React.FC = () => (
  <DeckFrame
    index={6}
    eyebrow="Live demo"
    title="Running in production."
    subtitle={`Deployed on Render and reachable at ${DECK_URL} — no MuJoCo required, the browser owns the 3D workcell.`}
    accent="cyan"
    dense
  >
    <div style={{display: "flex", gap: 26, height: "100%"}}>
      {[
        {src: "demo/dashboard.png", label: "Dashboard · live 3D workcell, command input and plan validation"},
        {src: "demo/refusal.png", label: "Safety refusal · unsupported destination blocked before motion"},
      ].map((shot) => (
        <div key={shot.src} style={{flex: 1, display: "flex", flexDirection: "column", gap: 14, justifyContent: "flex-start"}}>
          <div
            style={{
              border: `1px solid ${theme.lineStrong}`,
              borderRadius: 14,
              overflow: "hidden",
              background: theme.panel,
              lineHeight: 0,
            }}
          >
            <Img src={staticFile(shot.src)} style={{width: "100%", display: "block"}} />
          </div>
          <div style={{fontFamily: theme.mono, fontSize: 18, color: theme.muted, textAlign: "center"}}>
            {shot.label}
          </div>
        </div>
      ))}
    </div>
  </DeckFrame>
);

const sweepRows: Row[] = [
  {model: "policy", params: "552", fp32: "0.030 ms", int8: "0.031 ms", speedup: 0.97, bin: "0.70x"},
  {model: "vision", params: "134k", fp32: "0.040 ms", int8: "0.040 ms", speedup: 1.0, bin: "0.51x"},
  {model: "workcell-large", params: "5.0M", fp32: "0.152 ms", int8: "0.125 ms", speedup: 1.21, bin: "0.50x", highlight: true},
  {model: "workcell-xlarge", params: "19.9M", fp32: "0.597 ms", int8: "0.350 ms", speedup: 1.71, bin: "0.50x", highlight: true},
];

const SlideIntel: React.FC = () => (
  <DeckFrame
    index={7}
    eyebrow="Intel track · OpenVINO"
    title="A real optimization path, measured."
    subtitle="A genuine OpenVINO IR, quantized to INT8 with NNCF and benchmarked on CPU."

    accent="cyan"
    dense
  >
    <div style={{display: "flex", gap: 34, height: "100%", alignItems: "flex-start"}}>
      <div style={{display: "flex", flexDirection: "column", gap: 20, flex: 1}}>
        <Terminal
          width={760}
          fontSize={18}
          startFrame={-200}
          lines={[
            {prefix: "$", text: "python3.13 scripts/convert_openvino.py"},
            {text: "IR written: policy_mlp.xml (8,857 B) + policy_mlp.bin (1,104 B)", color: theme.green},
            {prefix: "$", text: "python3.13 scripts/quantize_openvino.py"},
            {text: "INT8 IR · max abs diff 0.0474 (rel 5.67%) · CPU calibration", color: theme.green},
            {prefix: "$", text: "python3.13 scripts/benchmark_openvino.py --sweep --throughput"},
            {text: "async throughput 2,723 → 5,003 inferences/sec (1.84x)", color: theme.cyan},
          ]}
        />
        <div
          style={{
            border: `1px solid ${theme.amber}44`,
            borderRadius: 14,
            background: "rgba(34,25,8,0.5)",
            padding: "20px 26px",
            fontSize: 20,
            lineHeight: 1.5,
            color: "#f0dcb4",
          }}
        >
          <b style={{color: theme.amber}}>Honest limits:</b> INT8 wins once the model is compute-bound
          (1.21x–1.71x). The tiny workcell policy is dispatch-bound at ~30 µs, so INT8 is neutral there — and
          we say so. GPU/NPU acceleration stays <b>gated</b> until it runs on Intel Core Ultra hardware; no
          accelerator numbers are fabricated.
        </div>
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 22, alignItems: "center"}}>
        <BenchmarkTable rows={sweepRows} startFrame={-200} />
        <div style={{fontFamily: theme.mono, fontSize: 17, color: theme.muted, textAlign: "center"}}>
          FP32 vs INT8 · median of 5 · Apple host exposing CPU only
        </div>
      </div>
    </div>
  </DeckFrame>
);

const SlideStack: React.FC = () => (
  <DeckFrame
    index={8}
    eyebrow="Built for a summit-ready demo"
    title="Production stack."
    subtitle="Standard-library backend, browser physics, and a real Intel optimization toolchain."
    accent="blue"
  >
    <div style={{display: "flex", flexDirection: "column", gap: 40, paddingTop: 24}}>
      <div style={{display: "flex", gap: 22}}>
        <Card icon="robot" accent={theme.cyan} tag="Simulation" title="Python engine" body="Authoritative 60 Hz task state and validated motion plans." />
        <Card icon="ai" accent={theme.cyan} tag="Language" title="Groq" body="Bounded tool-call loop over ten validated workcell tools." />
        <Card icon="diamond" accent={theme.blue} tag="Visualisation" title="Three.js + Rapier3D" body="3D workcell and browser physics telemetry for contact confirmation." />
        <Card icon="chip" accent={theme.green} tag="Optimization" title="Intel OpenVINO + NNCF" body="Real IR export, INT8 post-training quantization, CPU benchmarks." />
      </div>
      <div style={{display: "flex", flexDirection: "column", gap: 16, alignItems: "center"}}>
        <StackChips
          chips={[
            "Python stdlib HTTP server",
            "Three.js",
            "Rapier3D",
            "Groq",
            "MuJoCo (optional)",
            "Intel OpenVINO",
            "NNCF INT8",
            "Render",
          ]}
          startFrame={-200}
        />
        <div style={{fontFamily: theme.mono, fontSize: 19, color: theme.cyan}}>{DECK_URL} · deployed on Render</div>
      </div>
    </div>
  </DeckFrame>
);

const SlideMarket: React.FC = () => (
  <DeckFrame
    index={9}
    eyebrow="Market and business value"
    title="A safe testbed for physical AI."
    subtitle="Illustrative commercial framing — assumptions are shown and labelled, not verified market research."
    accent="green"
    dense
  >
    <div style={{display: "flex", gap: 34, height: "100%"}}>
      <div style={{flex: 1, display: "flex", flexDirection: "column", gap: 16}}>
        <div style={{fontSize: 24, fontWeight: 700}}>Who buys, and why now</div>
        <Row_ n={1} title="Robotics integrators" body="Tabletop, kitchen and warehouse cells that need a validated command layer before hardware exists." accent={theme.green} />
        <Row_ n={2} title="Physical-AI researchers" body="A reproducible, safe simulator to evaluate policies without risking a real arm." accent={theme.green} />
        <Row_ n={3} title="Silicon and platform vendors" body="Reference applications that demonstrate inference optimization on their accelerators." accent={theme.green} />
        <div
          style={{
            marginTop: "auto",
            border: `1px solid ${theme.line}`,
            borderLeft: `3px solid ${theme.green}`,
            borderRadius: 12,
            padding: "16px 20px",
            fontFamily: theme.mono,
            fontSize: 17,
            color: theme.muted,
            lineHeight: 1.55,
          }}
        >
          ILLUSTRATIVE, unverified: 2,000 robotics teams × $6k/yr ≈ $12M ARR. Shown so the assumptions
          can be challenged — not presented as market research.
        </div>
      </div>
      <div style={{flex: 1, display: "flex", flexDirection: "column", gap: 14}}>
        <div style={{fontSize: 24, fontWeight: 700}}>Revenue model</div>
        <Card compact tag="Open core" title="Free simulator" body="MIT core drives adoption; the paid tier adds multi-cell management and hosted runs." />
        <Card compact tag="Hosted" title="Simulation CI" body="Cloud workcell runs that regression-test policies and tool loops on every commit." />
        <Card compact tag="Services" title="Optimization + integration" body="OpenVINO / NNCF tuning and hardware bring-up engagements." />
      </div>
    </div>
  </DeckFrame>
);

const SlideClose: React.FC = () => (
  <DeckFrame
    index={10}
    eyebrow="Competition, roadmap, team"
    title="Language interprets intent; deterministic systems own execution."
    accent="cyan"
    dense
  >
    <div style={{display: "flex", gap: 30, height: "100%"}}>
      <div style={{flex: 1.15, display: "flex", flexDirection: "column", gap: 16}}>
        <div style={{fontSize: 25, fontWeight: 700}}>How VoxHands differs</div>
        <div
          style={{
            border: `1px solid ${theme.line}`,
            borderRadius: 14,
            overflow: "hidden",
            fontFamily: theme.mono,
            fontSize: 18,
          }}
        >
          {[
            ["Capability", "VoxHands", "Typical VLA demo"],
            ["Intent vs execution", "Split & enforced", "End-to-end policy"],
            ["Refusal semantics", "Explicit, blocks motion", "Usually absent"],
            ["State authority", "Server owns truth", "Model output"],
            ["Accelerator claims", "Verified or gated", "Often unstated"],
          ].map((row, i) => (
            <div
              key={row[0]}
              style={{
                display: "grid",
                gridTemplateColumns: "1.1fr 1fr 1fr",
                padding: "14px 18px",
                borderBottom: i === 4 ? "none" : `1px solid ${theme.line}`,
                color: i === 0 ? theme.faint : theme.text,
                background: i === 0 ? "rgba(10,22,33,0.8)" : "transparent",
              }}
            >
              <span>{row[0]}</span>
              <span style={{color: i === 0 ? theme.faint : theme.cyan}}>{row[1]}</span>
              <span style={{color: i === 0 ? theme.faint : theme.muted}}>{row[2]}</span>
            </div>
          ))}
        </div>
        <div
          style={{
            marginTop: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 10,
            fontFamily: theme.mono,
            fontSize: 21,
          }}
        >
          <span style={{color: theme.cyan}}>▸ Live demo · {DECK_URL}</span>
          <span style={{color: theme.muted}}>▸ Source · github.com/DevHuang1/VoxHands</span>
        </div>
      </div>
      <div style={{flex: 1, display: "flex", flexDirection: "column", gap: 22}}>
        <div style={{fontSize: 25, fontWeight: 700}}>Roadmap</div>
        <div style={{display: "flex", flexDirection: "column", gap: 14, fontSize: 20, color: theme.muted, lineHeight: 1.5}}>
          <div><b style={{color: theme.cyan}}>Now</b> — validated tool loop, authoritative simulator, real OpenVINO IR + INT8, measured CPU inference.</div>
          <div><b style={{color: theme.cyan}}>Next</b> — Intel Core Ultra NPU/GPU benchmarks published, real SO-101 bring-up, closed-loop vision policies.</div>
          <div><b style={{color: theme.cyan}}>Later</b> — ROS 2 bridge, time-coordinated control, hardware-partner integrations.</div>
        </div>
        <div style={{borderLeft: `3px solid ${theme.cyan}`, paddingLeft: 20, marginTop: 4}}>
          <div style={{fontSize: 22, fontWeight: 700}}>Team</div>
          <div style={{marginTop: 8, fontSize: 20, color: theme.muted}}>
            Thu Ta Sitt — Lead Developer
            <br />
            Yan Myo Thuyako — Software Engineer
          </div>
        </div>
      </div>
    </div>
  </DeckFrame>
);

/* --------------------------------------------------------------- assembly */

const deckSlides: React.FC[] = [
  SlideTitle,
  SlideProblem,
  SlideSolution,
  SlidePipeline,
  SlideSafety,
  SlideDemo,
  SlideIntel,
  SlideStack,
  SlideMarket,
  SlideClose,
];

export const DECK_SLIDES = deckSlides.length;

export const Deck: React.FC = () => {
  const frame = useCurrentFrame();
  const index = Math.max(0, Math.min(frame, deckSlides.length - 1));
  const Slide = deckSlides[index];
  return (
    <AbsoluteFill style={{background: theme.bg}}>
      <Slide />
    </AbsoluteFill>
  );
};
