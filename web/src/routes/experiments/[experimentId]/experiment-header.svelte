<script lang="ts">
  import { Globe, GlobeLock } from "@lucide/svelte";
  import type { Experiment } from "$lib/types";

  let { experiment }: { experiment: Experiment } = $props();

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }
</script>

<!-- Header -->
<div
  class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
>
  <div class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit">
    <div class="w-2 bg-ctp-blue rounded-full flex-shrink-0 self-stretch"></div>
    <div class="min-w-0 flex-1 py-1">
      <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
        {experiment.name}
      </h1>
      <div class="text-sm text-ctp-subtext0 space-y-1">
        <div>
          {#if experiment.description}
            <span>{experiment.description}</span>
          {/if}
        </div>
        <button
          onclick={() => copyToClipboard(experiment.id)}
          class="text-ctp-blue hover:text-ctp-blue/80 transition-colors flex text-start"
          title="click to copy experiment id"
        >
          <span>{experiment.id}</span>
        </button>
      </div>
    </div>
  </div>
</div>
