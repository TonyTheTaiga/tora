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
  <!-- Header -->
  <div class="flex justify-between items-center mb-3">
    <h3 class="font-medium text-lg text-ctp-text truncate pr-4">
      {experiment.name}
    </h3>
    <div class="flex flx-col space-x-2">
      <div
        class="p-1.5"
        class:text-ctp-green={experiment.visibility === "PUBLIC"}
        class:text-ctp-red={experiment.visibility === "PRIVATE"}
      >
        {#if experiment.visibility === "PUBLIC"}
          <Globe size={16} />
        {:else}
          <GlobeLock size={16} />
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
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
        title="Show experiment chain"
      >
        {#if highlighted.includes(experiment.id)}
          <EyeClosed size={16} />
        {:else}
          <Eye size={16} />
        {/if}
      </button>
      {#if page.data.user && page.data.user.id === experiment.user_id}
        <form method="POST" action="?/delete" class="flex items-center">
          <input type="hidden" name="id" value={experiment.id} />
          <button
            type="submit"
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-red"
            aria-label="Delete"
          >
            <X size={16} />
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
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
        aria-label="Expand details"
      >
        <Maximize2 size={16} />
      </button>
    </div>
  </div>

  <!-- Description -->
  {#if experiment.description}
    <p class="text-ctp-subtext0 text-sm leading-relaxed mb-4 line-clamp-2">
      {experiment.description}
    </p>
  {/if}

  <!-- Metrics indicator -->
  {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div class="flex items-center gap-1.5 text-ctp-subtext1 text-xs mb-3">
      <ChartLine size={14} />
      <span
        >{experiment.availableMetrics.length} metric{experiment.availableMetrics
          .length !== 1
          ? "s"
          : ""} available</span
      >
    </div>
  {/if}

  <!-- Footer info -->
  <div
    class="flex flex-wrap items-center justify-between gap-2 mt-auto pt-2 border-t border-ctp-surface0"
  >
    <!-- Tags -->
    {#if experiment.tags && experiment.tags.length > 0}
      <div class="flex items-center gap-1.5 text-xs text-ctp-subtext0">
        <Tag size={12} />
        {#each experiment.tags.slice(0, 2) as tag, i}
          <span class="px-1.5 py-0.5 bg-ctp-surface0 text-ctp-lavender">
            {tag}
          </span>
          {#if i === 0 && experiment.tags.length > 2}
            <span class="text-ctp-subtext0">+{experiment.tags.length - 1}</span>
          {/if}
        {/each}
      </div>
    {/if}

    <!-- Created At -->
    {#if experiment?.createdAt}
      <time class="flex items-center gap-1 text-xs text-ctp-subtext0">
        <Clock size={12} />
        {new Date(experiment.createdAt).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
        })}
      </time>
    {/if}
  </div>
</Card>
