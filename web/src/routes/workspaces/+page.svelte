<script lang="ts">
  let { data } = $props();
  let workspaces = $state(data.workspaces);
  $inspect(workspaces);
</script>

<div class="flex flex-col h-full">
  <div class="border-b border-ctp-mantle">
    <h1>Workspaces</h1>
  </div>
  <div class="flex-1">
    {#each workspaces as workspace, idx}
      <div class="flex flex-col">
        <h3>{workspace.name}</h3>
      </div>
    {/each}
  </div>
</div>
