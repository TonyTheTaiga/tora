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
    <div class="grid grid-cols-[auto,1fr,auto] items-center">
      <button
        class="col-start-1 flex flex-row space-x-1 items-center"
        aria-label={isMaximized ? "minimized" : "maximize"}
        onclick={() => {
          isMaximized = !isMaximized;
        }}
      >
        <div class="bg-ctp-overlay2 rounded-full w-2 h-2"></div>
        <div class="bg-ctp-overlay2 rounded-full w-2 h-2"></div>
        <div class="bg-ctp-blue rounded-full w-2 h-2"></div>
      </button>
      <p class="col-start-2 justify-self-center text-center">~/tora</p>
      <button
        class="col-start-3 bg-ctp-blue/30 border border-ctp-blue text-ctp-text justify-self-end"
        aria-label="signup"
        onclick={() => {
          goto("/signup");
        }}>create account</button
      >
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

<style lang="postcss">
  @reference "tailwindcss";
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

  /* Man-page style layout for README (borrowed from landing-page.svelte) */
  .readme {
    @apply text-xs sm:text-sm md:text-base leading-relaxed text-left;
    color: var(--color-ctp-text);
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .readme :global(h1) {
    @apply text-lg font-bold mb-4 font-mono;
    color: var(--color-ctp-text);
  }

  .readme :global(h2) {
    @apply text-base font-bold mb-3 mt-6 font-mono;
    color: var(--color-ctp-blue);
  }

  .readme :global(h3) {
    @apply text-sm font-bold mb-2 mt-4 font-mono;
    color: var(--color-ctp-mauve);
  }

  .readme :global(p) {
    @apply mb-3;
    color: var(--color-ctp-text);
  }

  .readme :global(ul) {
    @apply mb-3 pl-4;
  }

  .readme :global(li) {
    @apply mb-1;
    color: var(--color-ctp-text);
  }

  .readme :global(strong) {
    @apply font-bold;
    color: var(--color-ctp-text);
  }

  .readme :global(li strong:first-child) {
    color: var(--color-ctp-mauve);
  }

  .readme :global(em) {
    @apply italic;
    color: var(--color-ctp-subtext1);
  }

  .readme :global(code) {
    @apply px-1 py-0.5 rounded font-mono text-xs;
    background-color: rgba(var(--color-ctp-surface0), 0.3);
    color: var(--color-ctp-green);
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .readme :global(pre) {
    @apply font-mono text-xs p-3 rounded mb-3;
    background-color: rgba(var(--color-ctp-surface0), 0.3);
    color: var(--color-ctp-text);
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .readme :global(pre code) {
    @apply p-0;
    background-color: transparent;
    white-space: pre-wrap;
  }

  .readme :global(hr) {
    @apply my-6;
    border-color: var(--color-ctp-surface0);
  }

  .readme :global(blockquote) {
    @apply border-l-2 pl-4 italic;
    border-color: var(--color-ctp-blue);
    color: var(--color-ctp-subtext1);
  }
</style>
