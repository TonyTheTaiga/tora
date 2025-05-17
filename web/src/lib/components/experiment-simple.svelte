<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    Maximize2,
    Tag,
    Clock,
    ChartLine,
    Eye,
    EyeClosed,
    X,
    Globe,
    GlobeLock,
  } from "lucide-svelte";
  import Card from "./card.svelte";
  import { page } from "$app/state";

  let {
    experiment,
    selectedId = $bindable(),
    highlighted = $bindable(),
  }: {
    experiment: Experiment;
    selectedId: string | null;
    highlighted: string[];
  } = $props();
</script>

<Card
  background="bg-ctp-mantle"
  opacity={highlighted.length > 0 && !highlighted.includes(experiment.id)
    ? "opacity-40"
    : "opacity-100"}
  hover={highlighted.length > 0 && !highlighted.includes(experiment.id)
    ? false
    : true}
>
  <!-- Content wrapper with flex column and full height -->
  <div class="flex flex-col h-full">
    <!-- Header -->
    <div class="flex justify-between items-center mb-2">
      <h3 class="font-medium text-base text-ctp-text truncate pr-3">
        {experiment.name}
      </h3>
      <div class="flex items-center gap-1">
        <div
          class="p-1"
          class:text-ctp-green={experiment.visibility === "PUBLIC"}
          class:text-ctp-red={experiment.visibility === "PRIVATE"}
        >
          {#if experiment.visibility === "PUBLIC"}
            <Globe size={14} />
          {:else}
            <GlobeLock size={14} />
          {/if}
        </div>
        <button
          onclick={async () => {
            if (highlighted.includes(experiment.id)) {
              highlighted = [];
            } else {
              try {
                const response = await fetch(
                  `/api/experiments/${experiment.id}/ref`,
                );
                if (!response.ok) {
                  return;
                }
                const data = await response.json();
                highlighted = [...data, experiment.id];
              } catch (err) {}
            }
          }}
          class="p-1 text-ctp-subtext0 hover:text-ctp-text"
          title="Show experiment chain"
        >
          {#if highlighted.includes(experiment.id)}
            <EyeClosed size={14} />
          {:else}
            <Eye size={14} />
          {/if}
        </button>
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <form method="POST" action="?/delete" class="flex items-center">
            <input type="hidden" name="id" value={experiment.id} />
            <button
              type="submit"
              class="p-1 text-ctp-subtext0 hover:text-ctp-red"
              aria-label="Delete"
              title="Delete experiment"
            >
              <X size={14} />
            </button>
          </form>
        {/if}
        <button
          onclick={() => {
            if (selectedId === experiment.id) {
              selectedId = null;
            } else {
              selectedId = experiment.id;
            }
          }}
          class="p-1 text-ctp-subtext0 hover:text-ctp-text"
          aria-label="Expand details"
          title="Expand details"
        >
          <Maximize2 size={14} />
        </button>
      </div>
    </div>

    <!-- Middle content (grows to fill space) -->
    <div class="flex-grow flex flex-col">
      <!-- Description -->
      {#if experiment.description}
        <p class="text-ctp-subtext0 text-xs leading-relaxed mb-3 line-clamp-2">
          {experiment.description}
        </p>
      {/if}

      <!-- Metrics indicator -->
      <div class="flex items-center gap-1 text-ctp-subtext1 text-xs">
        <ChartLine size={12} />
        <span
          >{experiment.availableMetrics.length} metric{experiment.availableMetrics
            .length !== 1
            ? "s"
            : ""}</span
        >
      </div>
      
      <!-- Spacer that grows to push footer to bottom -->
      <div class="flex-grow"></div>
    </div>

    <!-- Footer info -->
    <div
      class="flex flex-wrap items-center justify-between gap-2 pt-1.5 border-t border-ctp-surface0 mt-2"
    >
      <!-- Tags -->
      {#if experiment.tags && experiment.tags.length > 0}
        <div class="flex items-center gap-1 text-xs text-ctp-subtext0">
          <Tag size={10} />
          {#each experiment.tags.slice(0, 2) as tag, i}
            <span class="px-1.5 py-0.5 bg-ctp-surface0/50 text-ctp-lavender rounded-full text-[10px]">
              {tag}
            </span>
            {#if i === 0 && experiment.tags.length > 2}
              <span class="text-ctp-subtext0 text-[10px]">+{experiment.tags.length - 1}</span>
            {/if}
          {/each}
        </div>
      {/if}

      <!-- Created At -->
      {#if experiment?.createdAt}
        <time class="flex items-center gap-0.5 text-[10px] text-ctp-subtext0">
          <Clock size={10} />
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
          })}
        </time>
      {/if}
    </div>
  </div>
</Card>
