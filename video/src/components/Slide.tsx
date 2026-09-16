import React from "react";
import {AbsoluteFill, Easing, interpolate, useCurrentFrame} from "remotion";
import {Accent, accentColor, theme} from "../theme";

export type SlideProps = {
  eyebrow: string;
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  caption: string;
  index: number;
  total: number;
  accent?: Accent;
  overallProgress: number;
  durationInFrames: number;
  children?: React.ReactNode;
};

const clamp = {extrapolateLeft: "clamp", extrapolateRight: "clamp"} as const;

export const Slide: React.FC<SlideProps> = ({
  eyebrow,
  title,
  subtitle,
  caption,
  index,
  total,
  accent = "cyan",
  overallProgress,
  durationInFrames,
  children,
}) => {
  const frame = useCurrentFrame();
  const color = accentColor(accent);

  const opacity = interpolate(
    frame,
    [0, 8, durationInFrames - 8, durationInFrames],
    [0, 1, 1, 0],
    clamp
  );
  const rise = interpolate(frame, [0, 18], [22, 0], {
    ...clamp,
    easing: Easing.out(Easing.cubic),
  });
  const visualOpacity = interpolate(frame, [4, 22], [0, 1], clamp);
  const visualRise = interpolate(frame, [4, 26], [30, 0], {
    ...clamp,
    easing: Easing.out(Easing.cubic),
  });

  return (
    <AbsoluteFill style={{fontFamily: theme.font, opacity, color: theme.text}}>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `radial-gradient(circle at 82% -6%, ${hexFade(color, 0.16)} 0%, transparent 34%), linear-gradient(180deg, #06101a 0%, ${theme.bg} 60%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          top: 0,
          height: 6,
          background: `linear-gradient(90deg, ${color}, transparent 70%)`,
          opacity: 0.5,
        }}
      />

      <div style={{position: "absolute", left: 150, top: 150, width: 880, transform: `translateY(${rise}px)`}}>
        <div
          style={{
            fontSize: 20,
            fontWeight: 700,
            letterSpacing: "0.34em",
            textTransform: "uppercase",
            color: color,
            opacity: 0.85,
          }}
        >
          {eyebrow}
        </div>
        <h1
          style={{
            margin: "34px 0 0",
            fontSize: 104,
            lineHeight: 0.98,
            letterSpacing: "-0.045em",
            fontWeight: 800,
            maxWidth: 900,
          }}
        >
          {title}
        </h1>
        {subtitle ? (
          <p
            style={{
              margin: "30px 0 0",
              fontSize: 32,
              lineHeight: 1.4,
              color: theme.muted,
              maxWidth: 760,
              fontWeight: 400,
            }}
          >
            {subtitle}
          </p>
        ) : null}
        <div style={{marginTop: 44, fontSize: 22, letterSpacing: "0.16em", color: theme.faint, fontFamily: theme.mono}}>
          {String(index).padStart(2, "0")} / {String(total).padStart(2, "0")}
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          right: 150,
          top: 0,
          bottom: 0,
          width: 760,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: visualOpacity,
          transform: `translateY(${visualRise}px)`,
        }}
      >
        {children}
      </div>

      <div
        style={{
          position: "absolute",
          left: 150,
          right: 150,
          bottom: 118,
          display: "flex",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            background: "rgba(3,10,16,0.82)",
            border: `1px solid ${theme.line}`,
            borderRadius: 14,
            padding: "20px 40px",
            fontSize: 26,
            fontWeight: 600,
            color: "#eaf3f6",
            textAlign: "center",
            maxWidth: 1080,
          }}
        >
          {caption}
        </div>
      </div>

      <div style={{position: "absolute", left: 150, right: 150, bottom: 76, height: 3, background: "rgba(255,255,255,0.09)", borderRadius: 99}}>
        <div
          style={{
            width: `${Math.max(0, Math.min(1, overallProgress)) * 100}%`,
            height: "100%",
            borderRadius: 99,
            background: `linear-gradient(90deg, ${color}, ${theme.blue})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};

const hexFade = (hex: string, alpha: number): string => {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};
