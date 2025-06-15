<script lang="ts">
  import type { Experiment } from "$lib/types";

  let { data } = $props();
  let experiments: Experiment[] = $derived(data.experiments);

  let searchQuery = $state("");

  let filteredExperiments = $derived(
    experiments.filter(
      (exp) =>
        exp.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (exp.description &&
          exp.description.toLowerCase().includes(searchQuery.toLowerCase())) ||
        (exp.tags &&
          exp.tags.some((tag) =>
            tag.toLowerCase().includes(searchQuery.toLowerCase()),
          )),
    ),
  );

  function formatDate(date: Date): string {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }
</script>

<div class="bg-ctp-base font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-6 border-b border-ctp-surface0/10"
  >
    <div class="flex items-center gap-4">
      <div class="w-2 h-8 bg-ctp-green rounded-full"></div>
      <div>
        <h1 class="text-xl font-bold text-ctp-text">public experiments</h1>
        <div class="text-xs text-ctp-subtext0">shared by the community</div>
      </div>
    </div>
  </div>

  <!-- Search bar -->
  <div class="px-6 py-4">
    <div class="max-w-lg">
      <div class="relative">
        <input
          type="text"
          placeholder="search experiments..."
          bind:value={searchQuery}
          class="w-full bg-ctp-surface0/20 border-0 px-4 py-3 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-text/20 transition-all font-mono text-sm"
        />
        <div
          class="absolute right-3 top-1/2 transform -translate-y-1/2 text-xs text-ctp-subtext0 font-mono"
        >
          {filteredExperiments.length}/{experiments.length}
        </div>
      </div>
    </div>
  </div>

  <!-- Experiment list -->
  <div class="px-6">
    {#if filteredExperiments.length === 0 && searchQuery}
      <div class="text-ctp-subtext0 text-sm">
        <div>$ search "{searchQuery}"</div>
        <div class="text-ctp-subtext1 ml-2">no results found</div>
      </div>
    {:else}
      <div class="space-y-1">
        {#each filteredExperiments as experiment}
          <a
            href={experiment.workspaceId
              ? `/workspaces/${experiment.workspaceId}/experiments/${experiment.id}`
              : "#"}
            class="group flex items-center gap-2 px-1 py-2 hover:bg-ctp-surface0/20 transition-colors"
          >
            <div class="text-ctp-green w-1">•</div>
            <div class="flex-1 min-w-0">
              <div class="flex items-center gap-2">
                <span
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
                >
                  {experiment.name}
                </span>
                {#if experiment.description}
                  <span class="text-ctp-subtext1 text-xs truncate"
                    >- {experiment.description}</span
                  >
                {/if}
              </div>
              {#if experiment.tags && experiment.tags.length > 0}
                <div class="flex gap-1 flex-wrap mt-1">
                  {#each experiment.tags.slice(0, 3) as tag}
                    <span
                      class="text-[10px] bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 px-1 py-px"
                    >
                      {tag}
                    </span>
                  {/each}
                  {#if experiment.tags.length > 3}
                    <span class="text-[10px] text-ctp-subtext0"
                      >+{experiment.tags.length - 3}</span
                    >
                  {/if}
                </div>
              {/if}
            </div>
            <div class="text-xs text-ctp-subtext0 flex-shrink-0">
              {formatDate(experiment.createdAt)}
            </div>
          </a>
        {/each}
      </div>
    {/if}
  </div>
</div>
