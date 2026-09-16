import React from "react";
import {Easing, interpolate, useCurrentFrame} from "remotion";
import {theme} from "../theme";

export type Stat = {value: string; label: string; color?: string};

const clamp = {extrapolateLeft: "clamp", extrapolateRight: "clamp"} as const;

export const Stats: React.FC<{items: Stat[]; startFrame?: number}> = ({items, startFrame = 8}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: "flex", gap: 26, flexDirection: "column", width: 640}}>
      {items.map((item, i) => {
        const delay = startFrame + i * 10;
        const appear = interpolate(frame, [delay, delay + 16], [0, 1], {
          ...clamp,
          easing: Easing.out(Easing.cubic),
        });
        return (
          <div
            key={item.label}
            style={{
              opacity: appear,
              transform: `translateY(${(1 - appear) * 22}px)`,
              borderLeft: `3px solid ${item.color ?? theme.cyan}`,
              paddingLeft: 24,
            }}
          >
            <div
              style={{
                fontFamily: theme.font,
                fontSize: 80,
                fontWeight: 800,
                letterSpacing: "-0.04em",
                lineHeight: 1,
                color: item.color ?? theme.text,
              }}
            >
              {item.value}
            </div>
            <div style={{marginTop: 10, fontSize: 22, color: theme.muted, fontWeight: 500}}>{item.label}</div>
          </div>
        );
      })}
    </div>
  );
};

export const StackChips: React.FC<{chips: string[]; startFrame?: number}> = ({chips, startFrame = 8}) => {
  const frame = useCurrentFrame();
  return (
    <div style={{display: "flex", flexWrap: "wrap", gap: 14, width: 620, justifyContent: "center"}}>
      {chips.map((chip, i) => {
        const delay = startFrame + i * 5;
        const appear = interpolate(frame, [delay, delay + 12], [0, 1], clamp);
        return (
          <span
            key={chip}
            style={{
              opacity: appear,
              transform: `translateY(${(1 - appear) * 14}px)`,
              border: `1px solid ${theme.lineStrong}`,
              borderRadius: 999,
              padding: "14px 26px",
              fontSize: 24,
              fontFamily: theme.mono,
              color: theme.text,
              background: "rgba(10,22,33,0.8)",
            }}
          >
            {chip}
          </span>
        );
      })}
    </div>
  );
};
