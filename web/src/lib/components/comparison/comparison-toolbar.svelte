<script lang="ts">
  import { getExperimentsSelectedForComparision } from "./state.svelte.js";
  import { Merge } from "lucide-svelte";

  let experiments = $derived(getExperimentsSelectedForComparision());
</script>

<div
  class="flex flex-row justify-between text-ctp-mantle rounded bg-ctp-teal mb-4 p-2 max-w-fit space-x-8"
>
  <span>{experiments.length} Selected</span>
  <button>
    <div class="flex flex-row items-center space-x-2">
      <span>Compare</span>
      <Merge size={16} />
    </div>
  </button>
</div>
