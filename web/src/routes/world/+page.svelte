<script lang="ts">
  let { data } = $props();
  let experiments = $derived(data.experiments);
  $inspect(experiments);
</script>
