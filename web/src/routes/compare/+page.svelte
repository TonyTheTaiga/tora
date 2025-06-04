<script lang="ts">
  import type { PageData } from "./$types";
  let { data }: { data: PageData } = $props();
  let experiments = $derived(data.experiments);
  $inspect(experiments);
</script>

<div class="text-ctp-text">Hello</div>
