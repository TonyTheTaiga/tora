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

<section aria-label="Terminal" class="min-h-0 flex flex-col">
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

  <div class="flex-1 min-h-0 overflow-y-auto overscroll-contain">
    {#if activeTab === "start"}
      <div>
        {@html highlightedCode}
      </div>
    {:else if activeTab === "readme"}
      <div>
        {@html processedUserGuide}
      </div>
    {/if}
  </div>
</section>
