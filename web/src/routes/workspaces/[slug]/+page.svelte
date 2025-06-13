<script lang="ts">
  import ExperimentsList from "$lib/components/experiments-list.svelte";
  let { data } = $props();
  let { experiments, workspace } = $derived(data);
</script>

<ExperimentsList {experiments} />
