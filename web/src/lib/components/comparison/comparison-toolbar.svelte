<script lang="ts">
  import { getExperimentsSelectedForComparision } from "./state.svelte.js";

  let experiments = $derived(getExperimentsSelectedForComparision());
</script>

<div class="flex flex-row text-ctp-text space-x-2">
  <span class="text-ctp-text">{experiments.length}</span>
  {#each experiments as experiment}
    <p>{experiment}</p>
  {/each}
</div>
