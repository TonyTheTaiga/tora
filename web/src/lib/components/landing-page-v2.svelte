<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
  import { browser } from "$app/environment";

  interface Props {
    highlightedCode: string;
    processedUserGuide: string;
  }

  let { highlightedCode, processedUserGuide }: Props = $props();

  let activeTab: "start" | "readme" = $state<"start" | "readme">("readme");
  let isMaximized = $state(false);
  $inspect(isMaximized);
  $inspect(activeTab);

  const headline = "Pure Speed. Pure Insight.";
</script>

<div class="fill-ctp-blue">
  <Logo />
</div>

<section class="font-mono">
  <h1>{headline}</h1>
</section>

<section aria-label="Terminal" class="min-h-0 min-w-0 flex flex-col">
  <header class="shrink-0 sticky top-0">
    <div class="grid grid-cols-3 items-center">
      <button
        class="flex flex-row space-x-1 items-center"
        aria-label={isMaximized ? "minimized" : "maximize"}
        onclick={() => {
          isMaximized = !isMaximized;
        }}
      >
        <div class="bg-ctp-overlay2 rounded-full w-2 h-2"></div>
        <div class="bg-ctp-overlay2 rounded-full w-2 h-2"></div>
        <div class="bg-ctp-blue rounded-full w-2 h-2"></div>
      </button>

      <p class="text-center">~/tora</p>
    </div>
  </header>

  <div class="shrink-0 sticky top-0 grid grid-cols-2">
    <button
      class="text-center"
      onclick={() => {
        activeTab = "start";
      }}>quick_start.txt</button
    >
    <button
      class="text-center"
      onclick={() => {
        activeTab = "readme";
      }}>README.md</button
    >
  </div>

  <div class="flex-1 min-h-0 min-w-0 overflow-y-auto overflow-x-hidden">
    {#if activeTab === "start"}
      <div class="quick_start">
        {@html highlightedCode}
      </div>
    {:else if activeTab === "readme"}
      <div class="readme">
        {@html processedUserGuide}
      </div>
    {/if}
  </div>
</section>

<style>
  /* Ensure terminal content fits horizontally on small screens
     Applied to classes defined in this component */
  :global(.quick_start pre),
  :global(.readme pre) {
    white-space: pre-wrap; /* wrap long lines while preserving breaks */
    overflow-wrap: anywhere; /* allow wrapping long tokens/URLs */
    word-break: break-word;
  }

  :global(.quick_start pre code),
  :global(.readme pre code) {
    white-space: inherit; /* match wrapping behavior of pre */
  }

  :global(.quick_start code),
  :global(.readme code) {
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  /* Make README/Quick Start tables and cells wrap instead of overflowing */
  :global(.quick_start table),
  :global(.readme table) {
    table-layout: fixed;
    width: 100%;
  }

  :global(.quick_start th),
  :global(.readme th),
  :global(.quick_start td),
  :global(.readme td) {
    white-space: normal;
    word-break: break-word;
  }

  /* Constrain images inside README/Quick Start */
  :global(.quick_start img),
  :global(.readme img) {
    max-width: 100%;
    height: auto;
  }

  /* Long links should wrap gracefully */
  :global(.quick_start a),
  :global(.readme a) {
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  /* Responsive font sizing for mobile */
  @media (max-width: 640px) {
    .quick_start,
    .readme {
      font-size: 0.92rem; /* slightly smaller on small screens */
      line-height: 1.4;
    }
    :global(.quick_start pre),
    :global(.quick_start code),
    :global(.readme pre),
    :global(.readme code) {
      font-size: 0.92em; /* scale with parent */
      line-height: 1.4;
    }
  }

  @media (max-width: 380px) {
    .quick_start,
    .readme {
      font-size: 0.85rem; /* tighter for very narrow phones */
    }
    :global(.quick_start pre),
    :global(.quick_start code),
    :global(.readme pre),
    :global(.readme code) {
      font-size: 0.9em;
    }
  }
</style>
