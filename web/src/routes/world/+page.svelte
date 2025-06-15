<script lang="ts">
  import PublicExperiments from "$lib/components/public-experiments.svelte";
  let { data } = $props();
</script>

<PublicExperiments experiments={data.experiments} />
