<script>
  import { onMount } from "svelte";
  import { Moon, Sun } from "lucide-svelte";

  let theme = "dark";

  onMount(() => {
    // Check for saved theme preference or use device preference
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
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

  function applyTheme(newTheme) {
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
  class="flex items-center justify-center w-9 h-9 rounded-md text-ctp-text hover:bg-ctp-surface0 transition-colors"
  aria-label={theme === "dark"
    ? "Switch to light theme"
    : "Switch to dark theme"}
  title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
>
  {#if theme === "dark"}
    <Sun size={20} />
  {:else}
    <Moon size={20} />
  {/if}
</button>
