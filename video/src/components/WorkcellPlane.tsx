import React from "react";
import {interpolate, useCurrentFrame} from "remotion";
import {theme} from "../theme";

const CX = 380;
const CY = 96;
const SX = 300;
const SY = 156;

const iso = (u: number, v: number) => ({
  x: CX + (u - v) * SX,
  y: CY + (u + v) * SY,
});

const GRID = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

type Highlight = "none" | "plate" | "cup" | "barrier" | "arms";

export const WorkcellPlane: React.FC<{highlight?: Highlight}> = ({highlight = "none"}) => {
  const frame = useCurrentFrame();
  const pulse = 0.55 + 0.45 * Math.sin(frame / 12);
  const plate = iso(0.31, 0.46);
  const cup = iso(0.62, 0.4);

  const holo = (active: boolean) => (active ? `0 0 ${8 + pulse * 10}px ${theme.cyan}` : "none");

  return (
    <svg viewBox="0 0 760 520" width={720} height={492} style={{overflow: "visible"}}>
      <defs>
        <linearGradient id="plane" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#123244" stopOpacity="0.75" />
          <stop offset="100%" stopColor="#07131d" stopOpacity="0.9" />
        </linearGradient>
      </defs>

      <polygon
        points={`${iso(0, 0).x},${iso(0, 0).y} ${iso(1, 0).x},${iso(1, 0).y} ${iso(1, 1).x},${iso(1, 1).y} ${iso(0, 1).x},${iso(0, 1).y}`}
        fill="url(#plane)"
        stroke={theme.lineStrong}
        strokeWidth={1.4}
      />
      <g stroke="rgba(120,170,195,0.14)" strokeWidth={1}>
        {GRID.map((t) => {
          const a = iso(t, 0);
          const b = iso(t, 1);
          const c = iso(0, t);
          const d = iso(1, t);
          return (
            <g key={t}>
              <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} />
              <line x1={c.x} y1={c.y} x2={d.x} y2={d.y} />
            </g>
          );
        })}
      </g>

      <line
        x1={iso(0.12, 0.62).x}
        y1={iso(0.12, 0.62).y}
        x2={iso(0.88, 0.34).x}
        y2={iso(0.88, 0.34).y}
        stroke={highlight === "barrier" ? theme.red : "rgba(220,90,80,0.72)"}
        strokeWidth={highlight === "barrier" ? 12 : 9}
        strokeLinecap="round"
        style={{filter: highlight === "barrier" ? `drop-shadow(0 0 ${10 + pulse * 12}px ${theme.red})` : "none"}}
      />

      <g stroke={theme.text} strokeWidth={13} strokeLinecap="round" opacity={0.9}
         style={{filter: highlight === "arms" ? `drop-shadow(0 0 ${10 + pulse * 12}px ${theme.cyan})` : "none"}}>
        <line x1={iso(0.36, 0.05).x} y1={iso(0.36, 0.05).y} x2={iso(0.46, 0.42).x} y2={iso(0.46, 0.42).y} />
        <line x1={iso(0.6, 0.03).x} y1={iso(0.6, 0.03).y} x2={iso(0.54, 0.34).x} y2={iso(0.54, 0.34).y} />
      </g>
      <circle cx={iso(0.46, 0.42).x} cy={iso(0.46, 0.42).y} r={7} fill={theme.cyan} />
      <circle cx={iso(0.54, 0.34).x} cy={iso(0.54, 0.34).y} r={7} fill={theme.cyan} />

      <ellipse cx={plate.x} cy={plate.y} rx={52} ry={25} fill="#3f7fc9" stroke="#8fc0ff" strokeWidth={1.5}
               style={{filter: highlight === "plate" ? hold(theme.blue, pulse) : "none"}} />
      <ellipse cx={plate.x} cy={plate.y - 6} rx={40} ry={17} fill="#5d9be0" opacity={0.65} />

      <ellipse cx={cup.x} cy={cup.y} rx={40} ry={19} fill="#d9cdae" stroke="#f3ead2" strokeWidth={1.5}
               style={{filter: highlight === "cup" ? hold(theme.amber, pulse) : "none"}} />
      <ellipse cx={cup.x} cy={cup.y - 9} rx={40} ry={19} fill="#efe4c6" />
      <ellipse cx={cup.x} cy={cup.y - 9} rx={30} ry={13} fill="#b9a882" opacity={0.7} />
    </svg>
  );
};

const hold = (color: string, pulse: number) => `drop-shadow(0 0 ${8 + pulse * 12}px ${color})`;
