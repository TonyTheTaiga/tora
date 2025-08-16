import { browser } from "$app/environment";

type Theme = "light" | "dark";

interface ThemeState {
  theme: Theme;
  systemPreference: Theme | "no-preference";
}

let state = $state<ThemeState>({
  theme: "dark",
  systemPreference: "no-preference",
});

function getSystemTheme(): Theme | "no-preference" {
  if (!browser) return "no-preference";

  if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
    return "dark";
  } else if (window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "no-preference";
}

function getStoredTheme(): Theme | null {
  if (!browser) return null;

  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") {
    return stored;
  }
  return null;
}

function setStoredTheme(theme: Theme) {
  if (!browser) return;
  localStorage.setItem("theme", theme);
}

function applyThemeToDOM(theme: Theme) {
  if (!browser) return;

  const html = document.documentElement;
  html.classList.remove("light", "dark");
  html.classList.add(theme);
  // Inform CSS functions like light-dark() of the active scheme
  // so client-side highlighted code (Shiki) renders correct colors.
  // Supported by modern browsers; harmless fallback otherwise.
  (html.style as any).colorScheme = theme;
}

function initializeTheme() {
  if (!browser) return;

  const storedTheme = getStoredTheme();
  const systemTheme = getSystemTheme();

  state.systemPreference = systemTheme;

  if (storedTheme) {
    state.theme = storedTheme;
  } else if (systemTheme !== "no-preference") {
    state.theme = systemTheme;
  } else {
    state.theme = "dark";
  }

  applyThemeToDOM(state.theme);

  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  mediaQuery.addEventListener("change", (e) => {
    state.systemPreference = e.matches ? "dark" : "light";
    if (!getStoredTheme()) {
      const newTheme = e.matches ? "dark" : "light";
      state.theme = newTheme;
      applyThemeToDOM(newTheme);
    }
  });
}

export function getTheme() {
  return state.theme;
}

export function getSystemPreference() {
  return state.systemPreference;
}

export function setTheme(theme: Theme) {
  state.theme = theme;
  setStoredTheme(theme);
  applyThemeToDOM(theme);
}

export function toggleTheme() {
  const newTheme = state.theme === "light" ? "dark" : "light";
  setTheme(newTheme);
}

export function useSystemTheme() {
  if (!browser) return;

  localStorage.removeItem("theme");

  if (state.systemPreference !== "no-preference") {
    state.theme = state.systemPreference;
    applyThemeToDOM(state.theme);
  }
}

export function isUsingSystemTheme() {
  return !getStoredTheme();
}

if (browser) {
  initializeTheme();
}
