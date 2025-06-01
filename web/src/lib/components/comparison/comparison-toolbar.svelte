<script lang="ts">
  import { getExperimentsSelectedForComparision } from "./state.svelte.js";
  import { Merge } from "lucide-svelte";
  import { goto } from "$app/navigation";

  let experiments = $derived(getExperimentsSelectedForComparision());
</script>

<div
  class="flex flex-row justify-between text-ctp-mantle rounded bg-ctp-teal mb-4 p-2 max-w-fit space-x-8"
>
  <span class="px-2 bg-ctp-overlay2/40 rounded"
    >{experiments.length} Selected</span
  >
  <button
    onclick={() => {
      goto("/compare");
    }}
  >
    <div
      class="flex flex-row items-center space-x-2 hover:bg-ctp-overlay2/40 rounded px-2"
    >
      <span>Compare</span>
      <Merge size={16} />
    </div>
  </button>
</div>
