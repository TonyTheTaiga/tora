<script lang="ts">
  import type {
    Experiment,
    ExperimentAnalysis,
    HPRecommendation,
  } from "$lib/types";
  import {
    Minimize2,
    X,
    Clock,
    Tag,
    Settings,
    Pencil,
    Info,
    ChartLine,
    Eye,
    EyeClosed,
    Sparkle,
    ClipboardCheck,
    Copy,
    ChevronDown,
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";
  import { page } from "$app/state";

  let {
    experiment = $bindable(),
    selectedId = $bindable(),
    highlighted = $bindable(),
    selectedForDelete = $bindable(),
    recentlyMinimized = $bindable(),
    selectedForEdit = $bindable(),
  }: {
    experiment: Experiment;
    selectedId: string | null;
    highlighted: string[];
    selectedForDelete: Experiment | null;
    recentlyMinimized: string | null;
    selectedForEdit: Experiment | null;
  } = $props();

  let recommendations = $state<Record<string, HPRecommendation>>({});
  let activeRecommendation = $state<string | null>(null);
  let idCopied = $state<boolean>(false);
</script>

<article
  class="bg-ctp-base overflow-hidden shadow-lg rounded-lg {highlighted.length >
    0 && !highlighted.includes(experiment.id)
    ? 'opacity-40'
    : ''}"
>
  <!-- Header with actions -->
  <header class="px-3 sm:px-4 py-3 bg-ctp-mantle border-b border-ctp-surface0">
    <!-- Mobile header (flex column) -->
    <div class="flex flex-col sm:hidden w-full gap-2">
      <!-- Title row -->
      <h2 class="truncate">
        <span
          role="button"
          tabindex="0"
          class="text-base font-medium cursor-pointer transition-all duration-150 flex items-center gap-1.5"
          class:text-ctp-green={idCopied}
          class:text-ctp-text={!idCopied}
          onclick={() => {
            navigator.clipboard.writeText(experiment.id);
            idCopied = true;
            setTimeout(() => {
              idCopied = false;
            }, 800);
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") e.currentTarget.click();
          }}
          title="Click to copy ID"
        >
          {#if idCopied}
            <span class="flex items-center">
              <ClipboardCheck size={16} class="mr-1 animate-bounce" />
              ID Copied
            </span>
          {:else}
            <span class="flex items-center">
              {experiment.name}
              <Copy size={12} class="ml-1 opacity-30 flex-shrink-0" />
            </span>
          {/if}
        </span>
      </h2>

      <!-- Actions row -->
      <div class="flex items-center justify-end gap-2">
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
            onclick={async () => {
              const response = await fetch(
                `/api/ai/analysis?experimentId=${experiment.id}`,
              );
              const data = (await response.json()) as ExperimentAnalysis;
              recommendations = data.hyperparameter_recommendations;
            }}
            title="Get AI recommendations"
          >
            <Sparkle size={16} />
          </button>
          <button
            onclick={() => {
              selectedForEdit = experiment;
            }}
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
            title="Edit experiment"
          >
            <Pencil size={16} />
          </button>
        {/if}
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
                // Ensure we don't add duplicate IDs and add the current experiment ID
                // This will show all experiments in the reference chain including the current one
                const uniqueIds = [...new Set([...data, experiment.id])];
                highlighted = uniqueIds;
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
          <input type="hidden" name="id" value={experiment.id} />
          <button
            type="button"
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-red"
            aria-label="Delete"
            title="Delete experiment"
            onclick={(e) => {
              e.stopPropagation();
              selectedForDelete = experiment;
            }}
          >
            <X size={16} />
          </button>
        {/if}
        <button
          onclick={() => {
            const currentId = experiment.id;
            // Set recently minimized to highlight the card
            recentlyMinimized = currentId;

            // First collapse
            selectedId = null;

            // Then scroll to the card's position
            setTimeout(() => {
              const element = document.getElementById(
                `experiment-${currentId}`,
              );
              if (element) {
                element.scrollIntoView({
                  behavior: "smooth",
                  block: "center",
                });
              }
            }, 10);
          }}
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
          aria-label="Minimize"
          title="Minimize"
        >
          <Minimize2 size={16} />
        </button>
      </div>
    </div>

    <!-- Desktop/Tablet header (flex row) -->
    <div class="hidden sm:flex sm:flex-row justify-between items-center">
      <h2 class="max-w-[70%]">
        <span
          role="button"
          tabindex="0"
          class="text-lg font-medium cursor-pointer transition-all duration-150 flex items-center gap-1.5"
          class:text-ctp-green={idCopied}
          class:text-ctp-text={!idCopied}
          onclick={() => {
            navigator.clipboard.writeText(experiment.id);
            idCopied = true;
            setTimeout(() => {
              idCopied = false;
            }, 800);
          }}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") e.currentTarget.click();
          }}
          title="Click to copy ID"
        >
          {#if idCopied}
            <span class="flex items-center">
              <ClipboardCheck size={18} class="mr-1 animate-bounce" />
              ID Copied
            </span>
          {:else}
            <span class="flex items-center truncate">
              <span class="truncate">{experiment.name}</span>
              <Copy size={14} class="ml-1 opacity-30 flex-shrink-0" />
            </span>
          {/if}
        </span>
      </h2>
      <div class="flex items-center gap-2">
        {#if page.data.user && page.data.user.id === experiment.user_id}
          <button
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
            onclick={async () => {
              const response = await fetch(
                `/api/ai/analysis?experimentId=${experiment.id}`,
              );
              const data = (await response.json()) as ExperimentAnalysis;
              recommendations = data.hyperparameter_recommendations;
            }}
            title="Get AI recommendations"
          >
            <Sparkle size={16} />
          </button>
          <button
            onclick={() => {
              selectedForEdit = experiment;
            }}
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
            title="Edit experiment"
          >
            <Pencil size={16} />
          </button>
        {/if}
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
                // Ensure we don't add duplicate IDs and add the current experiment ID
                // This will show all experiments in the reference chain including the current one
                const uniqueIds = [...new Set([...data, experiment.id])];
                highlighted = uniqueIds;
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
          <input type="hidden" name="id" value={experiment.id} />
          <button
            type="button"
            class="p-1.5 text-ctp-subtext0 hover:text-ctp-red"
            aria-label="Delete"
            title="Delete experiment"
            onclick={(e) => {
              e.stopPropagation();
              selectedForDelete = experiment;
            }}
          >
            <X size={16} />
          </button>
        {/if}
        <button
          onclick={() => {
            const currentId = experiment.id;
            // Set recently minimized to highlight the card
            recentlyMinimized = currentId;

            // First collapse
            selectedId = null;

            // Then scroll to the card's position
            setTimeout(() => {
              const element = document.getElementById(
                `experiment-${currentId}`,
              );
              if (element) {
                element.scrollIntoView({
                  behavior: "smooth",
                  block: "center",
                });
              }
            }, 10);
          }}
          class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
          aria-label="Minimize"
          title="Minimize"
        >
          <Minimize2 size={16} />
        </button>
      </div>
    </div>
  </header>

  <!-- Content Area -->
  <div class="px-2 sm:px-4 py-3 flex flex-col gap-3">
    <!-- Metadata section -->
    <div
      class="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4 text-ctp-subtext0 text-xs"
    >
      <div class="flex items-center gap-1">
        <Clock size={14} class="flex-shrink-0" />
        <time>
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
          })}
        </time>
      </div>

      {#if experiment.tags && experiment.tags.length > 0}
        <div
          class="flex items-center gap-1 overflow-x-auto sm:flex-wrap pb-1 sm:pb-0"
        >
          <Tag size={14} class="flex-shrink-0" />
          <div class="flex gap-1 flex-nowrap sm:flex-wrap">
            {#each experiment.tags as tag}
              <span
                class="whitespace-nowrap inline-flex items-center px-1.5 py-0.5 text-xs bg-ctp-surface0/50 text-ctp-blue rounded-full"
              >
                {tag}
              </span>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    {#if experiment.description}
      <p
        class="text-ctp-text text-sm py-1.5 border-l-2 border-ctp-mauve pl-3 mt-1 leading-relaxed"
      >
        {experiment.description}
      </p>
    {/if}

    <!-- Parameters section -->
    {#if experiment.hyperparams}
      <details class="mt-2">
        <summary
          class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
        >
          <Settings size={16} class="text-ctp-mauve flex-shrink-0" />
          <span class="text-sm font-medium">Parameters</span>
          <ChevronDown size={16} class="ml-auto text-ctp-subtext0" />
        </summary>
        <div class="pt-2">
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {#each experiment.hyperparams as param}
              <div
                class="flex items-center bg-ctp-mantle p-2 rounded-md overflow-hidden"
              >
                <span
                  class="text-xs font-medium text-ctp-subtext1 truncate max-w-[40%]"
                  >{param.key}</span
                >
                <div class="flex-grow"></div>
                <span
                  class="text-xs text-ctp-text px-2 py-0.5 bg-ctp-surface0 rounded-sm truncate max-w-[40%]"
                  >{param.value}</span
                >
                {#if recommendations && recommendations[param.key]}
                  <button
                    class="flex-shrink-0 pl-1.5"
                    onclick={() => {
                      activeRecommendation =
                        recommendations[param.key].recommendation;
                    }}
                    aria-label="Show recommendation"
                    title="Show AI recommendation"
                  >
                    <Info
                      size={14}
                      class="text-ctp-subtext0 hover:text-ctp-lavender"
                    />
                  </button>
                {/if}
              </div>
            {/each}
          </div>

          {#if activeRecommendation}
            <div
              class="mt-3 p-3 bg-ctp-surface0/50 border border-ctp-lavender/30 rounded-md relative"
            >
              <button
                class="absolute top-1.5 right-1.5 text-ctp-subtext0 hover:text-ctp-text"
                onclick={() => (activeRecommendation = null)}
                aria-label="Close recommendation"
              >
                <X size={14} />
              </button>
              <h4 class="text-xs font-medium text-ctp-lavender mb-1.5">
                AI Recommendation
              </h4>
              <p class="text-xs text-ctp-text leading-relaxed">
                {activeRecommendation}
              </p>
            </div>
          {/if}
        </div>
      </details>
    {/if}

    <!-- Metrics section -->
    {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
      <details class="mt-1" open>
        <summary
          class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
        >
          <ChartLine size={16} class="text-ctp-blue" />
          <span class="text-sm font-medium">Metrics</span>
          <ChevronDown size={16} class="ml-auto text-ctp-subtext0" />
        </summary>
        <!-- Full width chart container -->
        <div class="pt-2 -mx-2 sm:-mx-4">
          <div class="px-1 sm:px-2 w-full overflow-x-auto">
            <InteractiveChart {experiment} />
          </div>
        </div>
      </details>
    {/if}
  </div>
</article>

<style>
  /* Animation for card expansion */
  @keyframes expand {
    from {
      opacity: 0.4;
      transform: scale(0.95);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }

  :global(.animate-expand) {
    animation: expand 0.3s ease-out forwards;
  }
</style>
