import React from "react";
import {interpolate, useCurrentFrame} from "remotion";
import {theme} from "../theme";

export type TermLine = {text: string; color?: string; prefix?: string};

const clamp = {extrapolateLeft: "clamp", extrapolateRight: "clamp"} as const;

export const Terminal: React.FC<{
  title?: string;
  lines: TermLine[];
  startFrame?: number;
  step?: number;
  width?: number;
  fontSize?: number;
}> = ({title = "voxhands — zsh", lines, startFrame = 6, step = 9, width = 700, fontSize = 19}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        width,
        borderRadius: 14,
        border: `1px solid ${theme.line}`,
        background: "rgba(4,12,19,0.94)",
        boxShadow: "0 30px 60px rgba(0,0,0,0.4)",
        overflow: "hidden",
        fontFamily: theme.mono,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "12px 16px",
          borderBottom: `1px solid ${theme.line}`,
          color: theme.faint,
          fontSize: 15,
        }}
      >
        <span style={{width: 11, height: 11, borderRadius: 99, background: "#ff5f57"}} />
        <span style={{width: 11, height: 11, borderRadius: 99, background: "#febc2e"}} />
        <span style={{width: 11, height: 11, borderRadius: 99, background: "#28c840"}} />
        <span style={{marginLeft: 8, letterSpacing: "0.05em"}}>{title}</span>
      </div>
      <div style={{padding: "20px 22px", minHeight: 300}}>
        {lines.map((line, i) => {
          const appear = interpolate(frame, [startFrame + i * step, startFrame + i * step + 6], [0, 1], clamp);
          return (
            <div
              key={`${line.text}-${i}`}
              style={{
                opacity: appear,
                fontSize,
                lineHeight: 1.75,
                color: line.color ?? theme.text,
                whiteSpace: "pre-wrap",
                transform: `translateX(${(1 - appear) * -8}px)`,
              }}
            >
              {line.prefix ? <span style={{color: theme.cyan}}>{line.prefix} </span> : null}
              {line.text}
            </div>
          );
        })}
        <span
          style={{
            display: "inline-block",
            width: 10,
            height: fontSize + 4,
            marginTop: 6,
            background: theme.cyan,
            opacity: Math.sin(frame / 4) > 0 ? 1 : 0.15,
            verticalAlign: "middle",
          }}
        />
      </div>
    </div>
  );
};
