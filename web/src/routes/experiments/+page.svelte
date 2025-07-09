<script lang="ts">
  import { onMount } from "svelte";
  import type { PageData } from "./$types";
  import { page } from "$app/stores";
  import { goto } from "$app/navigation";

  interface Experiment {
    id: string;
    name: string;
    description: string;
    hyperparams: any[];
    tags: string[];
    createdAt: Date;
    updatedAt: Date;
    availableMetrics: string[];
    workspaceId: string;
  }

  let { data } = $props();
  let { experiments, workspace } = $derived(data);

  let searchQuery = $state("");
  let sortBy = $state<"name" | "created" | "updated" | "metrics">("created");
  let sortOrder = $state<"asc" | "desc">("desc");

  let filteredAndSortedExperiments = $derived(() => {
    let filtered: Experiment[] = experiments;

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = experiments.filter((exp) => {
        const searchableText =
          `${exp.name} ${exp.description} ${exp.tags.join(" ")}`.toLowerCase();
        return query.split(" ").every((term) => searchableText.includes(term));
      });
    }

    // Sort experiments
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case "name":
          comparison = a.name.localeCompare(b.name);
          break;
        case "created":
          comparison = a.createdAt.getTime() - b.createdAt.getTime();
          break;
        case "updated":
          comparison = a.updatedAt.getTime() - b.updatedAt.getTime();
          break;
        case "metrics":
          comparison = a.availableMetrics.length - b.availableMetrics.length;
          break;
      }

      return sortOrder === "desc" ? -comparison : comparison;
    });

    return filtered;
  });

  const handleKeydown = (event: KeyboardEvent) => {
    if (event.key === "/") {
      event.preventDefault();
      const searchElement = document.querySelector<HTMLInputElement>(
        'input[type="search"]',
      );
      searchElement?.focus();
    }
  };

  onMount(() => {
    window.addEventListener("keydown", handleKeydown);
    return () => {
      window.removeEventListener("keydown", handleKeydown);
    };
  });

  function formatDate(date: Date): string {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    }).format(date);
  }

  function formatRelativeTime(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return formatDate(date);
    }
  }

  function toggleSort(field: typeof sortBy) {
    if (sortBy === field) {
      sortOrder = sortOrder === "asc" ? "desc" : "asc";
    } else {
      sortBy = field;
      sortOrder = "desc";
    }
  }

  function getSortIcon(field: typeof sortBy): string {
    if (sortBy !== field) return "";
    return sortOrder === "asc" ? "↑" : "↓";
  }
</script>

<div class="font-mono">
  <!-- Header -->
  <div
    class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
  >
    <div
      class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
    >
      <div
        class="w-2 bg-ctp-green rounded-full flex-shrink-0 self-stretch"
      ></div>
      <div class="min-w-0 flex-1 py-1">
        <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
          Experiments
          {#if workspace}
            <span class="text-ctp-subtext0">in workspace</span>
          {/if}
        </h1>
        <div class="text-sm text-ctp-subtext0 space-y-1">
          <div>
            {filteredAndSortedExperiments().length} experiment{filteredAndSortedExperiments()
              .length !== 1
              ? "s"
              : ""}
            {searchQuery ? `matching "${searchQuery}"` : ""}
          </div>
        </div>
      </div>
    </div>

    <div class="flex gap-2">
      {#if workspace}
        <button
          onclick={() => goto("/experiments")}
          class="bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-3 py-2 text-sm font-mono transition-all"
        >
          All Experiments
        </button>
      {/if}
      <button
        onclick={() => {
          /* TODO: Open create experiment modal */
        }}
        class="group relative bg-ctp-surface0/20 backdrop-blur-md border border-ctp-surface0/30 text-ctp-text hover:bg-ctp-surface0/30 hover:border-ctp-surface0/50 px-3 py-2 md:px-4 text-sm font-mono transition-all flex-shrink-0"
      >
        <div class="flex items-center gap-2">
          <svg
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 4v16m8-8H4"
            ></path>
          </svg>
          <span class="hidden sm:inline">New</span>
        </div>
      </button>
    </div>
  </div>

  <!-- Search and filters -->
  <div class="px-4 md:px-6 py-4 border-b border-ctp-surface0/10">
    <div class="flex flex-col sm:flex-row gap-4">
      <!-- Search -->
      <div class="flex-1 max-w-lg">
        <div
          class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
        >
          <span class="text-ctp-subtext0 font-mono text-sm px-4 py-3">/</span>
          <input
            type="search"
            placeholder="search experiments..."
            bind:value={searchQuery}
            class="flex-1 bg-transparent border-0 py-3 pr-4 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-base"
          />
        </div>
      </div>

      <!-- Sort controls -->
      <div class="flex items-center gap-2 text-sm">
        <span class="text-ctp-subtext0">Sort by:</span>
        <select
          bind:value={sortBy}
          class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-text px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-ctp-text/20"
        >
          <option value="created">Created</option>
          <option value="updated">Updated</option>
          <option value="name">Name</option>
          <option value="metrics">Metrics</option>
        </select>
        <button
          onclick={() => (sortOrder = sortOrder === "asc" ? "desc" : "asc")}
          class="text-ctp-subtext0 hover:text-ctp-text transition-colors px-2 py-1"
          title={`Sort ${sortOrder === "asc" ? "descending" : "ascending"}`}
        >
          {sortOrder === "asc" ? "↑" : "↓"}
        </button>
      </div>
    </div>
  </div>

  <!-- Experiments list -->
  <div class="px-4 md:px-6 font-mono">
    {#if filteredAndSortedExperiments().length === 0}
      <div class="text-center py-12">
        {#if searchQuery}
          <div class="text-ctp-subtext0 text-base">
            <div>No experiments found matching "{searchQuery}"</div>
            <button
              onclick={() => (searchQuery = "")}
              class="text-ctp-blue hover:text-ctp-sky transition-colors mt-2"
            >
              Clear search
            </button>
          </div>
        {:else}
          <div class="text-ctp-subtext0 text-base">
            <div>No experiments found</div>
            <div class="text-ctp-subtext1 mt-2">
              Create your first experiment to get started
            </div>
          </div>
        {/if}
      </div>
    {:else}
      <!-- Table header -->
      <div class="border-b border-ctp-surface0/20 py-2">
        <div class="grid grid-cols-12 gap-4 text-sm text-ctp-subtext0">
          <div class="col-span-1">•</div>
          <div class="col-span-4">
            <button
              onclick={() => toggleSort("name")}
              class="hover:text-ctp-text transition-colors"
            >
              name {getSortIcon("name")}
            </button>
          </div>
          <div class="col-span-2">
            <button
              onclick={() => toggleSort("metrics")}
              class="hover:text-ctp-text transition-colors"
            >
              metrics {getSortIcon("metrics")}
            </button>
          </div>
          <div class="col-span-2">tags</div>
          <div class="col-span-2">
            <button
              onclick={() => toggleSort("updated")}
              class="hover:text-ctp-text transition-colors"
            >
              updated {getSortIcon("updated")}
            </button>
          </div>
          <div class="col-span-1">actions</div>
        </div>
      </div>

      <!-- Experiments -->
      {#each filteredAndSortedExperiments() as experiment}
        <div
          class="group border-b border-ctp-surface0/10 last:border-0 hover:bg-ctp-surface0/10 transition-colors"
        >
          <div class="grid grid-cols-12 gap-4 py-3 items-center">
            <!-- Status indicator -->
            <div class="col-span-1">
              <div class="text-ctp-green text-sm">•</div>
            </div>

            <!-- Name and description -->
            <div class="col-span-4 min-w-0">
              <a
                href="/experiments/{experiment.id}"
                class="block min-w-0 group"
              >
                <div
                  class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate"
                >
                  {experiment.name}
                </div>
                {#if experiment.description}
                  <div class="text-ctp-subtext1 text-sm truncate mt-1">
                    {experiment.description}
                  </div>
                {/if}
              </a>
            </div>

            <!-- Metrics -->
            <div class="col-span-2">
              {#if experiment.availableMetrics.length > 0}
                <div class="text-ctp-text text-sm">
                  {experiment.availableMetrics.length} metric{experiment
                    .availableMetrics.length !== 1
                    ? "s"
                    : ""}
                </div>
                <div class="text-ctp-subtext1 text-xs truncate">
                  {experiment.availableMetrics.slice(0, 2).join(", ")}
                  {#if experiment.availableMetrics.length > 2}
                    +{experiment.availableMetrics.length - 2} more
                  {/if}
                </div>
              {:else}
                <div class="text-ctp-subtext1 text-sm">No metrics</div>
              {/if}
            </div>

            <!-- Tags -->
            <div class="col-span-2">
              {#if experiment.tags.length > 0}
                <div class="flex flex-wrap gap-1">
                  {#each experiment.tags.slice(0, 3) as tag}
                    <span
                      class="text-ctp-blue text-xs bg-ctp-surface0/30 px-1 rounded"
                    >
                      {tag}
                    </span>
                  {/each}
                  {#if experiment.tags.length > 3}
                    <span class="text-ctp-subtext1 text-xs">
                      +{experiment.tags.length - 3}
                    </span>
                  {/if}
                </div>
              {:else}
                <div class="text-ctp-subtext1 text-sm">—</div>
              {/if}
            </div>

            <!-- Updated -->
            <div class="col-span-2 text-ctp-subtext0 text-sm">
              {formatRelativeTime(experiment.updatedAt)}
            </div>

            <!-- Actions -->
            <div class="col-span-1">
              <div class="opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  class="text-ctp-subtext0 hover:text-ctp-red text-sm p-1"
                  title="Delete experiment"
                  onclick={(e) => {
                    e.preventDefault();
                    // TODO: Show delete confirmation modal
                    console.log("Delete experiment:", experiment.id);
                  }}
                >
                  ×
                </button>
              </div>
            </div>
          </div>
        </div>
      {/each}

      <!-- Summary -->
      <div
        class="flex items-center text-sm text-ctp-subtext0 pt-4 pb-2 border-t border-ctp-surface0/20"
      >
        <div class="flex-1">
          Showing {filteredAndSortedExperiments().length} of {experiments.length}
          experiment{experiments.length !== 1 ? "s" : ""}
        </div>
      </div>
    {/if}
  </div>
</div>
