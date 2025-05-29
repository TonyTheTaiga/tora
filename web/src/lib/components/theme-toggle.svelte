<script lang="ts">
  import { onMount } from "svelte";
  import { Moon, Sun } from "lucide-svelte";

  let theme: "dark" | "light" = "dark";

  onMount(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme && (savedTheme === "dark" || savedTheme === "light")) {
      theme = savedTheme;
      applyTheme(theme);
    } else {
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
  class="toolbar-button"
  aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
  title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
>
  {#if theme === "dark"}
    <Sun class="icon" />
  {:else}
    <Moon class="icon" />
  {/if}
</button>

<style>
  .toolbar-button {
    padding: 0.5rem;
    color: var(--color-ctp-subtext0);
    transition: color 0.2s, background-color 0.2s;
  }
  
  .toolbar-button:hover {
    color: var(--color-ctp-text);
    background-color: var(--color-ctp-surface2);
  }
  
  :global(.icon) {
    width: 20px;
    height: 20px;
  }
  
  /* Responsive styles */
  @media (min-width: 640px) {
    .toolbar-button {
      padding: 0.625rem;
    }
    
    :global(.icon) {
      width: 20px;
      height: 20px;
    }
  }
</style>
