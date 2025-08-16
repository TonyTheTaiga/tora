<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";

  interface Props {
    highlightedCode: string;
    processedUserGuide: string;
  }

  let { highlightedCode, processedUserGuide }: Props = $props();
  let activeTab: "start" | "readme" = $state<"start" | "readme">("readme");
  let isMaximized = $state(false);
  let windowControlHovered = $state(false);
  function isUserOnMobile() {
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    );
  }
  const headline = "Pure Speed. Pure Insight.";
  const subtitle = "A New Experiment Tracker";
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
        class="grid grid-cols-[auto,1fr,auto] items-center h-9 px-3 border-b border-ctp-surface0/60 bg-ctp-mantle/80 backdrop-blur"
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
          ~/tora
        </p>
        <button
          class="col-start-3 justify-self-end text-xs px-2 py-1 rounded border border-ctp-blue/60 bg-ctp-blue/20 text-ctp-text hover:bg-ctp-blue/30"
          aria-label="signin"
          onclick={() => {
            goto("/login");
          }}>sign in</button
        >
      </div>
    </header>

    <div
      class="shrink-0 sticky top-9 z-10 bg-ctp-base/95 backdrop-blur border-b border-ctp-surface0/60 flex"
    >
      <button
        class="px-4 py-2 text-xs font-mono text-ctp-subtext1 hover:text-ctp-text"
        class:text-ctp-text={activeTab === "start"}
        class:border-ctp-blue={activeTab === "start"}
        class:border-b-2={activeTab === "start"}
        onclick={() => {
          activeTab = "start";
        }}>quick_start.txt</button
      >
      <button
        class="px-4 py-2 text-xs font-mono text-ctp-subtext1 hover:text-ctp-text"
        class:text-ctp-text={activeTab === "readme"}
        class:border-ctp-blue={activeTab === "readme"}
        class:border-b-2={activeTab === "readme"}
        onclick={() => {
          activeTab = "readme";
        }}>README.md</button
      >
    </div>

    <div class="flex-1 min-h-0 min-w-0 overflow-y-auto overflow-x-hidden p-4">
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
</div>

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
