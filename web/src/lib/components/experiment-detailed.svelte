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
  } from "lucide-svelte";
  import InteractiveChart from "./interactive-chart.svelte";
  import EditExperimentModal from "./edit-experiment-modal.svelte";

  let {
    experiment = $bindable(),
    selectedId = $bindable(),
    highlighted = $bindable(),
  }: {
    experiment: Experiment;
    selectedId: string | null;
    highlighted: string[];
  } = $props();

  let editMode = $state<boolean>(false);
  let recommendations = $state<Record<string, HPRecommendation> | null>(null);
  let activeRecommendation = $state<string | null>(null);
  let idCopied = $state<boolean>(false);
  $inspect(experiment.availableMetrics);
</script>

{#if editMode}
  <EditExperimentModal bind:experiment bind:editMode />
{/if}

<article class="bg-ctp-base overflow-hidden shadow-lg">
  <!-- Header with actions -->
  <header
    class="p-4 bg-ctp-mantle border-b border-ctp-surface0 flex flex-col md:flex-row justify-between items-center"
  >
    <h2>
      <span
        role="button"
        tabindex="0"
        class="text-xl font-semibold content-center cursor-pointer transition-all duration-150 flex items-center gap-2"
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
            <ClipboardCheck size={20} class="mr-1 animate-bounce" />
            ID Copied
          </span>
        {:else}
          <span class="flex items-center">
            {experiment.name}
            <Copy size={14} class="ml-1 opacity-30" />
          </span>
        {/if}
      </span>
    </h2>
    <div class="flex items-center gap-3">
      <button
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
        onclick={async () => {
          const response = await fetch(
            `/api/ai/analysis?experimentId=${experiment.id}`,
          );
          const data = (await response.json()) as ExperimentAnalysis;
          recommendations = data.hyperparameter_recommendations;
        }}
      >
        <Sparkle size={16} />
      </button>
      <button
        onclick={() => {
          editMode = true;
        }}
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
      >
        <Pencil size={16} />
      </button>
      <button
        onclick={async () => {
          if (highlighted.at(-1) === experiment.id) {
            highlighted = [];
          } else {
            const response = await fetch(
              `/api/experiments/${experiment.id}/ref`,
            );
            const data = (await response.json()) as string[];
            highlighted = data;
          }
        }}
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
      >
        {#if highlighted.at(-1) === experiment.id}
          <EyeClosed size={16} />
        {:else}
          <Eye size={16} />
        {/if}
      </button>
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
      <button
        onclick={() => {
          if (selectedId === experiment.id) {
            selectedId = null;
          } else {
            selectedId = experiment.id;
          }
        }}
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
        aria-label="Minimize"
      >
        <Minimize2 size={16} />
      </button>
    </div>
  </header>

  <!-- Metadata section -->
  <div class="p-5 border-b border-ctp-surface0">
    <div class="flex items-center gap-6 mb-4 text-ctp-subtext0 text-sm">
      <div class="flex items-center gap-1.5">
        <Clock size={14} />
        <time>
          {new Date(experiment.createdAt).toLocaleDateString("en-US", {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "numeric",
          })}
        </time>
      </div>

      {#if experiment.tags && experiment.tags.length > 0}
        <div class="flex items-center gap-1.5">
          <Tag size={14} />
          <div class="flex flex-wrap gap-2">
            {#each experiment.tags as tag}
              <span
                class="inline-flex items-center px-2 py-0.5 text-xs bg-ctp-surface0 text-ctp-mauve"
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
        class="text-ctp-text text-sm py-2 border-l-2 border-ctp-mauve pl-3 my-3 max-w-prose leading-relaxed"
      >
        {experiment.description}
      </p>
    {/if}
  </div>

  <!-- Parameters section -->
  {#if experiment.hyperparams}
    <div class="p-5 border-b border-ctp-surface0">
      <div class="flex items-center gap-2 mb-4">
        <Settings size={16} class="text-ctp-mauve" />
        <h3 class="text-lg font-semibold text-ctp-mauve">Parameters</h3>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {#each experiment.hyperparams as param}
          <div class="flex items-center bg-ctp-mantle p-3">
            <span class="text-sm font-medium text-ctp-subtext1"
              >{param.key}</span
            >
            <div class="flex-grow"></div>
            <span class="text-sm text-ctp-text px-2 py-1 bg-ctp-surface0"
              >{param.value}</span
            >
            {#if recommendations && recommendations[param.key]}
              <button
                class="items-center pl-2"
                onclick={() =>
                  (activeRecommendation =
                    recommendations[param.key].recommendation)}
                aria-label="Show recommendation"
              >
                <Info
                  size={16}
                  class="text-ctp-subtext0 hover:text-ctp-lavender"
                />
              </button>
            {/if}
          </div>
        {/each}
      </div>

      {#if activeRecommendation}
        <div
          class="mt-5 p-4 bg-ctp-surface0 border border-ctp-lavender rounded-md relative"
        >
          <button
            class="absolute top-2 right-2 text-ctp-subtext0 hover:text-ctp-text"
            onclick={() => (activeRecommendation = null)}
            aria-label="Close recommendation"
          >
            <X size={16} />
          </button>
          <h4 class="text-sm font-medium text-ctp-lavender mb-2">
            AI Recommendation
          </h4>
          <p class="text-sm text-ctp-text leading-relaxed">
            {activeRecommendation}
          </p>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Metrics section -->
  {#if experiment.availableMetrics && experiment.availableMetrics.length > 0}
    <div>
      <div class="p-5 pb-0">
        <div class="flex items-center gap-2">
          <ChartLine size={16} class="text-ctp-mauve" />
          <h3 class="text-lg font-semibold text-ctp-mauve">Charts</h3>
        </div>
      </div>
      <InteractiveChart {experiment} />
    </div>
  {/if}

  <!-- AI Analysis section -->
  <!-- <ExperimentAiAnalysis {experiment} bind:aiSuggestions /> -->
</article>
