<script lang="ts">
  import { onMount } from "svelte";
  import { Moon, Sun } from "lucide-svelte";

  let theme: "dark" | "light" = "dark";

  onMount(() => {
    // Check for saved theme preference or use device preference
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme && (savedTheme === "dark" || savedTheme === "light")) {
      theme = savedTheme;
      applyTheme(theme);
    } else {
      // Use device preference if no saved theme
      const prefersDark = window.matchMedia(
        "(prefers-color-scheme: dark)",
      ).matches;
      theme = prefersDark ? "dark" : "light";
      applyTheme(theme);
    }
  });

  function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    localStorage.setItem("theme", theme);
  }

  function applyTheme(newTheme: "dark" | "light") {
    if (newTheme === "dark") {
      document.documentElement.classList.remove("light");
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
      document.documentElement.classList.add("light");
    }
  }
</script>

<button
  on:click={toggleTheme}
  class="p-1.5 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface2 transition-colors"
  aria-label={theme === "dark"
    ? "Switch to light theme"
    : "Switch to dark theme"}
  title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
>
  {#if theme === "dark"}
    <Sun size={16} />
  {:else}
    <Moon size={16} />
  {/if}
</button>
