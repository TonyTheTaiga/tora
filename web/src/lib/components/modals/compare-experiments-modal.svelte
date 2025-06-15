<script lang="ts">
  import { goto } from "$app/navigation";
  import { X } from "lucide-svelte";
  import type { Experiment } from "$lib/types";
  import { closeCompareExperimentsModal } from "$lib/state/app.svelte.js";

  let { experiments = $bindable(), currentId = $bindable() }:
    { experiments: Experiment[]; currentId: string } = $props();

  let selected = $state<string[]>([]);

  function toggle(id: string) {
    selected = selected.includes(id)
      ? selected.filter((v) => v !== id)
      : [...selected, id];
  }

  function compare() {
    const ids = [currentId, ...selected].join(",");
    goto(`/compare?ids=${ids}`);
    closeCompareExperimentsModal();
  }
</script>

<div class="fixed inset-0 bg-ctp-base/90 backdrop-blur-sm flex items-center justify-center p-4 z-50 font-mono">
  <div class="w-full max-w-md bg-ctp-base border border-ctp-surface0/30 max-h-[90vh] overflow-auto">
    <div class="flex items-center justify-between p-4 border-b border-ctp-surface0/20">
      <div class="flex items-center gap-3">
        <div class="w-2 h-6 bg-ctp-sapphire rounded-full"></div>
        <h3 class="text-lg text-ctp-text">compare experiments</h3>
      </div>
      <button
        onclick={closeCompareExperimentsModal}
        class="text-ctp-subtext0 hover:text-ctp-red rounded p-1"
      >
        <X size={14} />
      </button>
    </div>
    <div class="p-4 space-y-1">
      {#each experiments as exp}
        {#if exp.id !== currentId}
          <label class="flex items-center gap-2 p-1 hover:bg-ctp-surface0/20 cursor-pointer text-sm">
            <input
              type="checkbox"
              checked={selected.includes(exp.id)}
              onchange={() => toggle(exp.id)}
              class="text-ctp-blue focus:ring-ctp-blue focus:ring-1 w-3 h-3"
            />
            <span class="text-ctp-text truncate">{exp.name}</span>
          </label>
        {/if}
      {/each}
      {#if experiments.filter((e) => e.id !== currentId).length === 0}
        <div class="text-sm text-ctp-subtext0 text-center">no experiments found</div>
      {/if}
    </div>
    <div class="flex justify-end gap-2 p-4 border-t border-ctp-surface0/20">
      <button
        type="button"
        onclick={closeCompareExperimentsModal}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:text-ctp-text px-3 py-2 text-xs"
      >
        cancel
      </button>
      <button
        type="button"
        onclick={compare}
        class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:text-ctp-blue px-3 py-2 text-xs {selected.length === 0 ? 'opacity-50 cursor-not-allowed' : ''}"
        disabled={selected.length === 0}
      >
        compare
      </button>
    </div>
  </div>
</div>
