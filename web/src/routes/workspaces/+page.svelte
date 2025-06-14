<script lang="ts">
  let { data } = $props();
  let { workspaces } = $derived(data);
</script>

<div class="text-ctp-text">
  {#each workspaces as workspace}
    <div>
      <a href={`/workspaces/${workspace.id}`}>{workspace.name}</a>
    </div>
  {/each}
</div>
