import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {theme} from "../theme";

export type Row = {
  model: string;
  params: string;
  fp32: string;
  int8: string;
  speedup: number;
  bin: string;
  highlight?: boolean;
};

const clamp = {extrapolateLeft: "clamp", extrapolateRight: "clamp"} as const;

export const BenchmarkTable: React.FC<{rows: Row[]; startFrame?: number}> = ({rows, startFrame = 10}) => {
  const frame = useCurrentFrame();
  return (
    <div
      style={{
        width: 720,
        border: `1px solid ${theme.line}`,
        borderRadius: 16,
        background: "rgba(6,16,25,0.9)",
        padding: "22px 26px",
        fontFamily: theme.mono,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.5fr 1.1fr 0.9fr 0.9fr 1fr",
          fontSize: 16,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: theme.faint,
          paddingBottom: 14,
          borderBottom: `1px solid ${theme.line}`,
        }}
      >
        <span>model</span>
        <span style={{textAlign: "right"}}>params</span>
        <span style={{textAlign: "right"}}>fp32</span>
        <span style={{textAlign: "right"}}>int8</span>
        <span style={{textAlign: "right"}}>speedup</span>
      </div>
      {rows.map((row, i) => {
        const rowIn = interpolate(frame, [startFrame + i * 7, startFrame + i * 7 + 12], [0, 1], clamp);
        const barWidth = interpolate(
          interpolate(frame, [startFrame + i * 7 + 4, startFrame + i * 7 + 22], [0, 1], {
            ...clamp,
            easing: Easing.out(Easing.cubic),
          }),
          [0, 1],
          [0, Math.min(1, (row.speedup - 0.9) / 0.9)],
          clamp
        );
        const active = row.highlight;
        return (
          <div
            key={row.model}
            style={{
              display: "grid",
              gridTemplateColumns: "1.5fr 1.1fr 0.9fr 0.9fr 1fr",
              alignItems: "center",
              fontSize: 19,
              padding: "15px 0",
              borderBottom: i === rows.length - 1 ? "none" : `1px solid ${theme.line}`,
              opacity: rowIn,
              color: active ? theme.text : theme.muted,
            }}
          >
            <span style={{color: active ? theme.cyan : theme.text, fontWeight: active ? 700 : 400}}>{row.model}</span>
            <span style={{textAlign: "right"}}>{row.params}</span>
            <span style={{textAlign: "right"}}>{row.fp32}</span>
            <span style={{textAlign: "right"}}>{row.int8}</span>
            <span style={{display: "flex", alignItems: "center", gap: 10, justifyContent: "flex-end"}}>
              <span
                style={{
                  position: "relative",
                  width: 84,
                  height: 6,
                  borderRadius: 99,
                  background: "rgba(255,255,255,0.08)",
                  overflow: "hidden",
                }}
              >
                <span
                  style={{
                    position: "absolute",
                    inset: 0,
                    width: `${barWidth * 100}%`,
                    borderRadius: 99,
                    background: active ? theme.green : theme.blue,
                  }}
                />
              </span>
              <span style={{color: active ? theme.green : theme.muted, minWidth: 62, textAlign: "right"}}>
                {row.speedup.toFixed(2)}x
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
};
