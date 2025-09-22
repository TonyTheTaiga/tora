import { browser } from "$app/environment";

export const CHART_COLOR_KEYS = [
  "blue",
  "lavender",
  "sky",
  "green",
  "teal",
  "mauve",
  "peach",
  "yellow",
  "pink",
  "sapphire",
  "maroon",
  "red",
  "rosewater",
];

export type ChartTheme = {
  colors: string[];
  text: string;
  mantle: string;
  overlay0: string;
  sky: string;
  fadedGridLines: string;
  axisTicks: string;
  terminalBg?: string;
  terminalBorder?: string;
};

export function getChartTheme(): ChartTheme {
  if (!browser) {
    return {
      colors: ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"],
      text: "#111",
      mantle: "#fff",
      overlay0: "#888",
      sky: "#2aa1ff",
      fadedGridLines: "#ddd",
      axisTicks: "#666",
    };
  }

  const cs = getComputedStyle(document.documentElement);
  const colors = CHART_COLOR_KEYS.map((k) =>
    cs.getPropertyValue(`--color-ctp-${k}`).trim(),
  ).filter(Boolean);

  return {
    colors: colors.length ? colors : ["#4e79a7", "#f28e2b", "#e15759"],
    text: cs.getPropertyValue("--color-ctp-text").trim(),
    mantle: cs.getPropertyValue("--color-ctp-mantle").trim(),
    overlay0: cs.getPropertyValue("--color-ctp-overlay0").trim(),
    sky: cs.getPropertyValue("--color-ctp-sky").trim(),
    fadedGridLines: cs.getPropertyValue("--color-ctp-surface1").trim() + "33",
    axisTicks: cs.getPropertyValue("--color-ctp-subtext0").trim(),
    terminalBg: cs.getPropertyValue("--color-ctp-terminal-bg").trim(),
    terminalBorder: cs.getPropertyValue("--color-ctp-terminal-border").trim(),
  };
}
