export const theme = {
  bg: "#050b12",
  bgTop: "#0a1a26",
  panel: "#0b1621",
  panelSoft: "#0e1d2a",
  line: "rgba(156,190,207,0.16)",
  lineStrong: "rgba(156,190,207,0.30)",
  text: "#edf5f7",
  muted: "#8297a4",
  faint: "#536875",
  cyan: "#5ee1dc",
  blue: "#70a8ff",
  green: "#73dfa1",
  amber: "#e8b969",
  red: "#ff716d",
  font: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif',
  mono: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
} as const;

export type Accent = "cyan" | "blue" | "green" | "amber" | "red";

export const accentColor = (accent: Accent = "cyan"): string => theme[accent];
