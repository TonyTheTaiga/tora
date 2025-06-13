<script lang="ts">
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  let { data } = $props();
  let { experiments } = $derived(data);
</script>

<ExperimentsList {experiments} />
