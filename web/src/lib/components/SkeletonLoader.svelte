<script lang="ts">
  let {
    variant = "text",
    count = 1,
  }: {
    variant?: "text" | "card" | "list-item" | "chart";
    count?: number;
  } = $props();
</script>

{#if variant === "list-item"}
  {#each Array(count) as _, i}
    <div class="animate-pulse px-2 py-3">
      <div class="flex items-start justify-between gap-4">
        <div class="flex-1 space-y-2">
          <div class="h-4 bg-ctp-surface0/40 w-3/4"></div>
          <div class="h-3 bg-ctp-surface0/30 w-1/2"></div>
        </div>
        <div class="h-8 w-8 bg-ctp-surface0/30"></div>
      </div>
    </div>
  {/each}
{:else if variant === "card"}
  {#each Array(count) as _, i}
    <div class="animate-pulse p-4 border border-ctp-surface0/30">
      <div class="space-y-3">
        <div class="h-5 bg-ctp-surface0/40 w-2/3"></div>
        <div class="h-3 bg-ctp-surface0/30 w-full"></div>
        <div class="h-3 bg-ctp-surface0/30 w-4/5"></div>
      </div>
    </div>
  {/each}
{:else if variant === "chart"}
  <div class="animate-pulse h-80 w-full border border-ctp-surface0/20">
    <div class="h-full flex items-end gap-2 p-4">
      <div class="flex-1 bg-ctp-surface0/30 h-[40%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[65%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[35%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[80%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[55%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[45%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[70%]"></div>
      <div class="flex-1 bg-ctp-surface0/30 h-[30%]"></div>
    </div>
  </div>
{:else}
  {#each Array(count) as _, i}
    <div class="animate-pulse h-4 bg-ctp-surface0/40 w-full"></div>
  {/each}
{/if}
