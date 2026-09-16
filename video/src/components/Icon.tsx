import React from "react";
import {theme} from "../theme";

export type IconName =
  | "voice"
  | "check"
  | "ai"
  | "diamond"
  | "refresh"
  | "shield"
  | "chip"
  | "gauge"
  | "bolt"
  | "layers"
  | "boundary"
  | "robot";

export const Icon: React.FC<{name: IconName; color?: string; size?: number}> = ({
  name,
  color = theme.cyan,
  size = 130,
}) => {
  const common = {
    fill: "none",
    stroke: color,
    strokeWidth: 7,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  if (name === "ai") {
    return (
      <span
        style={{
          fontFamily: theme.font,
          fontSize: size * 0.86,
          fontWeight: 800,
          letterSpacing: "-0.04em",
          color,
        }}
      >
        AI
      </span>
    );
  }

  return (
    <svg viewBox="0 0 120 120" width={size} height={size}>
      {name === "voice" && (
        <g {...common}>
          <rect x={50} y={26} width={20} height={42} rx={10} />
          <path d="M38 58a22 22 0 0 0 44 0" />
          <line x1={60} y1={80} x2={60} y2={94} />
          <line x1={46} y1={94} x2={74} y2={94} />
        </g>
      )}
      {name === "check" && <path {...common} d="M30 62l20 20 40-46" />}
      {name === "diamond" && (
        <g {...common}>
          <path d="M60 20l28 28-28 44-28-44z" />
          <path d="M32 48h56" />
        </g>
      )}
      {name === "refresh" && (
        <g {...common}>
          <path d="M92 60a32 32 0 1 1-9-22" />
          <path d="M92 26v16H76" />
        </g>
      )}
      {name === "shield" && (
        <g {...common}>
          <path d="M60 18l32 12v24c0 22-14 36-32 46-18-10-32-24-32-46V30z" />
          <path d="M46 60l10 10 20-22" />
        </g>
      )}
      {name === "chip" && (
        <g {...common}>
          <rect x={34} y={34} width={52} height={52} rx={8} />
          <rect x={50} y={50} width={20} height={20} rx={3} />
          {[44, 60, 76].map((p) => (
            <React.Fragment key={p}>
              <line x1={p} y1={18} x2={p} y2={34} />
              <line x1={p} y1={86} x2={p} y2={102} />
              <line x1={18} y1={p} x2={34} y2={p} />
              <line x1={86} y1={p} x2={102} y2={p} />
            </React.Fragment>
          ))}
        </g>
      )}
      {name === "gauge" && (
        <g {...common}>
          <path d="M24 80a36 36 0 0 1 72 0" />
          <line x1={60} y1={80} x2={84} y2={54} />
          <circle cx={60} cy={80} r={6} />
        </g>
      )}
      {name === "bolt" && <path {...common} d="M66 18L38 66h20l-8 36 32-52H62z" />}
      {name === "layers" && (
        <g {...common}>
          <path d="M60 24l38 20-38 20-38-20z" />
          <path d="M22 62l38 20 38-20" />
        </g>
      )}
      {name === "boundary" && (
        <g {...common}>
          <rect x={22} y={30} width={76} height={60} rx={8} />
          <line x1={44} y1={30} x2={44} y2={90} />
          <line x1={76} y1={30} x2={76} y2={90} />
        </g>
      )}
      {name === "robot" && (
        <g {...common}>
          <rect x={34} y={42} width={52} height={44} rx={8} />
          <circle cx={50} cy={64} r={5} fill={color} />
          <circle cx={70} cy={64} r={5} fill={color} />
          <line x1={60} y1={42} x2={60} y2={28} />
          <circle cx={60} cy={24} r={5} />
        </g>
      )}
    </svg>
  );
};

export const IconBadge: React.FC<{name: IconName; color?: string; size?: number}> = ({
  name,
  color = theme.cyan,
  size = 300,
}) => (
  <div
    style={{
      width: size,
      height: size,
      borderRadius: "50%",
      border: `1.5px solid ${color}44`,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}
  >
    <Icon name={name} color={color} size={size * 0.44} />
  </div>
);
