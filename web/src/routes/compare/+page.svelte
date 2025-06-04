<script lang="ts">
  import { getExperimentsSelectedForComparision } from "$lib/components/comparison/state.svelte";
  let experiments = $derived(getExperimentsSelectedForComparision());
  $inspect(experiments);
</script>

<div class="text-ctp-text">Hello</div>
