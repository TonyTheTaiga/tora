<script lang="ts">
  import type { PageData } from "./$types";
  import type { Experiment } from "$lib/types";

  let { data }: { data: PageData } = $props();
  let { experiments }: { experiments: Experiment[] } = $derived(data);
  let query = $state("");
  let filtered = $derived(
    experiments.filter((e) => {
      const q = query.toLowerCase();
      return (
        e.name.toLowerCase().includes(q) ||
        (e.description?.toLowerCase() || "").includes(q) ||
        (e.tags || []).some((t) => t.toLowerCase().includes(q))
      );
    }),
  );
  function formatDate(date: Date) {
    return new Date(date).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "2-digit",
    });
  }
</script>

<div class="font-mono">
  <div class="p-4 space-y-4">
    <div class="max-w-lg">
      <input
        type="text"
        bind:value={query}
        placeholder="search experiments..."
        class="w-full bg-ctp-surface0/20 border-0 px-4 py-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-text/20 transition-all font-mono text-sm"
      />
    </div>
    <div class="space-y-2">
      {#each filtered as experiment}
        <a
          href={`/experiments/${experiment.id}`}
          class="block bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 p-3 hover:bg-ctp-surface0/20 transition-colors"
        >
          <div class="flex items-center justify-between">
            <div class="text-ctp-blue truncate">{experiment.name}</div>
            <div class="text-xs text-ctp-subtext0">
              {formatDate(experiment.createdAt)}
            </div>
          </div>
          {#if experiment.description}
            <div class="text-ctp-subtext1 text-xs line-clamp-2">
              {experiment.description}
            </div>
          {/if}
        </a>
      {/each}
    </div>
  </div>
</div>
