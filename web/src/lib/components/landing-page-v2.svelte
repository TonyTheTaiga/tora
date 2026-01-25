<script lang="ts">
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";
  import { gettingStartedContent } from "$lib/content";
  import Logo from "$lib/logo_assets/logo.svelte";

  let highlightedCode = $state<string>("");
  let isMaximized = $state(false);
  let windowControlHovered = $state(false);
  function isUserOnMobile() {
    if (typeof navigator === "undefined") return false;
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    );
  }
  const headline = "Pure Speed. Pure Insight.";
  const subtitle = "A New Experiment Tracker";

  onMount(async () => {
    try {
      const { codeToHtml } = await import("shiki");

      highlightedCode = await codeToHtml(gettingStartedContent, {
        lang: "python",
        themes: { dark: "catppuccin-mocha", light: "catppuccin-latte" },
        defaultColor: "light-dark()",
      });
    } catch (error) {
      console.error("Client-side content processing failed", error);
      const lines = gettingStartedContent.trim().split("\n");
      const fallbackCode = lines
        .map((line, i) => {
          const num = (i + 1).toString().padStart(2, " ");
          return `<span class=\"text-ctp-overlay0 select-none\">${num}</span>  ${line}`;
        })
        .join("\n");
      highlightedCode = `<pre class=\"text-ctp-text font-mono\"><code>${fallbackCode}</code></pre>`;
    }
  });
</script>

<div
  class="h-full w-full grid grid-rows-[auto,auto,1fr]"
  class:p-4={!isMaximized}
>
  {#if !isMaximized}
    <div
      class="fill-ctp-blue w-full max-w-2xs md:max-w-xl mx-auto mb-4 translate-x-[5.5%]"
    >
      <Logo />
    </div>

    <section class="font-mono mb-4 text-center text-ctp-text">
      <h1 class="text-xl md:text-4xl">{headline}</h1>
      <h2 class="text-lg md:text-2xl">{subtitle}</h2>
    </section>
  {/if}

  <section
    aria-label="Terminal"
    class="min-h-0 min-w-0 flex flex-col bg-ctp-base border border-ctp-surface0/60 rounded-xl shadow-lg overflow-hidden mx-auto w-full max-w-5xl"
  >
    <header class="shrink-0 sticky top-0 z-10">
      <div
        class="grid grid-cols-[auto,1fr,auto] items-center h-12 px-4 border-b border-ctp-surface0/60 bg-ctp-mantle/80 backdrop-blur"
      >
        <button
          class="col-start-1 flex flex-row items-center gap-2"
          aria-label={isMaximized ? "minimized" : "maximize"}
          onclick={() => {
            isMaximized = !isMaximized;
          }}
          onmouseenter={() => {
            windowControlHovered = true;
          }}
          onmouseleave={() => {
            windowControlHovered = false;
          }}
        >
          <div
            class="rounded-full w-3 h-3"
            class:bg-ctp-blue={windowControlHovered || isUserOnMobile()}
            class:bg-ctp-overlay2={!windowControlHovered && !isUserOnMobile()}
          ></div>
          <div
            class="rounded-full w-3 h-3"
            class:bg-ctp-blue={windowControlHovered || isUserOnMobile()}
            class:bg-ctp-overlay2={!windowControlHovered && !isUserOnMobile()}
          ></div>
          <div
            class="rounded-full w-3 h-3"
            class:bg-ctp-blue={windowControlHovered || isUserOnMobile()}
            class:bg-ctp-overlay2={!windowControlHovered && !isUserOnMobile()}
          ></div>
        </button>
        <p
          class="col-start-2 justify-self-center text-center text-xs text-ctp-subtext1"
        >
          ~/tora/quick_start.txt
        </p>
        <div class="col-start-3 justify-self-end flex items-center gap-2">
          <button
            class="text-[11px] font-mono text-ctp-overlay0 hover:text-ctp-text transition"
            aria-label="user guide"
            onclick={() => {
              goto("/api-docs/");
            }}>user guide</button
          >
          <button
            class="text-[11px] font-mono text-ctp-subtext1 hover:text-ctp-text transition"
            aria-label="signin"
            onclick={() => {
              goto("/login");
            }}>sign in</button
          >
        </div>
      </div>
    </header>

    <div class="flex-1 min-h-0 min-w-0 overflow-y-auto overflow-x-hidden p-4">
      <div class="quick_start">
        {@html highlightedCode}
      </div>
    </div>
  </section>
</div>

<style lang="postcss">
  @reference "tailwindcss";
  /* Ensure terminal content fits horizontally on small screens
     Applied to classes defined in this component */
  :global(.quick_start pre) {
    white-space: pre-wrap; /* wrap long lines while preserving breaks */
    overflow-wrap: anywhere; /* allow wrapping long tokens/URLs */
    word-break: break-word;
  }

  :global(.quick_start pre code) {
    white-space: inherit; /* match wrapping behavior of pre */
  }

  :global(.quick_start code) {
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  /* Make README/Quick Start tables and cells wrap instead of overflowing */
  :global(.quick_start table) {
    table-layout: fixed;
    width: 100%;
  }

  :global(.quick_start th),
  :global(.quick_start td) {
    white-space: normal;
    word-break: break-word;
  }

  /* Constrain images inside README/Quick Start */
  :global(.quick_start img) {
    max-width: 100%;
    height: auto;
  }

  /* Long links should wrap gracefully */
  :global(.quick_start a) {
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  /* Responsive font sizing for mobile */
  @media (max-width: 640px) {
    .quick_start {
      font-size: 0.92rem; /* slightly smaller on small screens */
      line-height: 1.4;
    }
    :global(.quick_start pre),
    :global(.quick_start code) {
      font-size: 0.92em; /* scale with parent */
      line-height: 1.4;
    }
  }

  @media (max-width: 380px) {
    .quick_start {
      font-size: 0.85rem; /* tighter for very narrow phones */
    }
    :global(.quick_start pre),
    :global(.quick_start code) {
      font-size: 0.9em;
    }
  }
</style>
