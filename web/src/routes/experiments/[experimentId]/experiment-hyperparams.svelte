<script lang="ts">
  import { Copy, ClipboardCheck } from "lucide-svelte";
  import type { HyperParam } from "$lib/types";

  let {
    hyperparams,
    initialLimit = 10,
    onCopyParam,
    copiedParam
  }: {
    hyperparams: HyperParam[];
    initialLimit?: number;
    onCopyParam: (value: string, key: string) => void;
    copiedParam: string | null;
  } = $props();

  let showAllParams = $state(false);

  let visibleParams = $derived(
    showAllParams || hyperparams.length <= initialLimit
      ? hyperparams
      : hyperparams.slice(0, initialLimit),
  );
</script>

{#if hyperparams && hyperparams.length > 0}
  <div class="space-y-2">
    <div class="flex items-center gap-2">
      <div class="text-sm text-ctp-text">hyperparameters</div>
      <div class="text-sm text-ctp-subtext0 font-mono">
        [{hyperparams.length}]
      </div>
    </div>
    <div class="bg-ctp-surface0/10 border border-ctp-surface0/20">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-0">
        {#each visibleParams as param}
          <div
            class="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-ctp-surface0/10 hover:bg-ctp-surface0/20 px-3 py-2 transition-colors text-sm gap-1 sm:gap-2"
          >
            <span class="text-ctp-subtext0 font-mono truncate"
              >{param.key}</span
            >
            <div class="flex items-center gap-2">
              <span
                class="text-ctp-blue font-mono bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 max-w-24 sm:max-w-32 truncate"
                title={String(param.value)}
              >
                {param.value}
              </span>
              <button
                onclick={() => onCopyParam(String(param.value), param.key)}
                class="text-ctp-subtext0 hover:text-ctp-text transition-colors"
              >
                {#if copiedParam === param.key}
                  <ClipboardCheck size={10} class="text-ctp-green" />
                {:else}
                  <Copy size={10} />
                {/if}
              </button>
            </div>
          </div>
        {/each}
      </div>
      {#if hyperparams.length > initialLimit}
        <button
          onclick={() => (showAllParams = !showAllParams)}
          class="w-full text-sm text-ctp-subtext0 hover:text-ctp-text px-3 py-2 text-center border-t border-ctp-surface0/20 transition-colors"
        >
          {showAllParams
            ? "show less"
            : `show ${hyperparams.length - initialLimit} more`}
        </button>
      {/if}
    </div>
  </div>
{/if}