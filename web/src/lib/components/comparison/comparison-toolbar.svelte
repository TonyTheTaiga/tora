<script lang="ts">
  import { getExperimentsSelectedForComparision } from "$lib/state/comparison.svelte.js";
  import { Merge } from "lucide-svelte";
  import { goto } from "$app/navigation";

  let experiments = $derived(getExperimentsSelectedForComparision());
</script>

<div
  class="flex flex-row justify-between items-center text-ctp-text rounded-2xl bg-ctp-surface1 shadow-lg py-2 px-4 w-auto sm:max-w-md gap-4" /* Pill styling */
>
  <span class="font-medium flex-shrink-0">{experiments.length} experiment{experiments.length === 1 ? '' : 's'} selected</span>
  <button
    onclick={() => {
      const params = experiments.join(",");
      goto(`/compare?ids=${params}`);
    }}
    class="flex flex-row items-center space-x-2 px-3 py-1.5 bg-ctp-blue text-ctp-base rounded-md hover:bg-ctp-blue/90 transition-colors" /* Button styling */
  >
    <span>Compare</span>
    <Merge size={16} /> {/* Slightly larger icon */}
  </button>
</div>
