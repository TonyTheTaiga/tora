<script lang="ts">
  import { getExperimentsSelectedForComparision } from "$lib/components/comparison/state.svelte";
  import { Merge } from "lucide-svelte";
  import { goto } from "$app/navigation";

  let experiments = $derived(getExperimentsSelectedForComparision());
</script>

<div
  class="flex flex-row justify-between text-ctp-mantle rounded bg-ctp-sky mb-4 p-2 max-w-fit space-x-8"
>
  <span class="px-2 rounded">{experiments.length} Selected</span>
  <button
    onclick={() => {
      const params = experiments.join(",");
      goto(`/compare?ids=${params}`);
    }}
  >
    <div class="flex flex-row items-center space-x-2 rounded px-2">
      <span>Compare</span>
      <Merge size={14} />
    </div>
  </button>
</div>
