<script lang="ts">
  import {
    X,
    Save,
    TagIcon,
    Plus,
    Link,
    Globe,
    Lock,
    ChevronDown,
  } from "lucide-svelte";
  import { enhance } from "$app/forms";
  import type { Experiment } from "$lib/types";
  import { onMount, onDestroy } from "svelte";
  import { closeEditExperimentModal } from "$lib/state/app.svelte.js";

  let {
    experiment = $bindable(),
    experiments = $bindable(),
  }: {
    experiment: Experiment;
    experiments: Experiment[];
  } = $props();

  let experimentCopy = $state<Experiment>({
    id: experiment.id,
    name: experiment.name,
    description: experiment.description,
    visibility: experiment.visibility,
    tags: experiment.tags ? [...experiment.tags] : [],
    availableMetrics: experiment.availableMetrics
      ? [...experiment.availableMetrics]
      : [],
    hyperparams: experiment.hyperparams ? [...experiment.hyperparams] : [],
    createdAt: experiment.createdAt,
    updatedAt: experiment.updatedAt,
  });

  let addingNewTag = $state(false);
  let tag = $state<string | null>(null);
  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");

  let filteredExperiments = $derived(
    experiments.filter(
      (exp) =>
        exp.id !== experiment.id &&
        exp.name.toLowerCase().includes(searchInput.toLowerCase()),
    ),
  );

  onMount(async () => {
    document.body.classList.add("overflow-hidden");

    try {
      reference = null;
      const response = await fetch(`/api/experiments/${experiment.id}/ref`);
      if (response.ok) {
        const referenceIds = await response.json();
        const referencesToLoad = referenceIds.filter(
          (id: String) => id !== experiment.id,
        );

        if (referencesToLoad.length > 0) {
          try {
            const refResponse = await fetch(
              `/api/experiments/${referencesToLoad[0]}`,
            );
            if (refResponse.ok) {
              reference = await refResponse.json();
            }
          } catch (refError) {
            console.error("Failed to load referenced experiment:", refError);
          }
        }
      }
    } catch (error) {
      console.error("Failed to load references:", error);
    }
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });

  function selectReference(exp: Experiment) {
    reference = exp;
  }

  function clearReference() {
    reference = null;
  }

  function clearSearch() {
    searchInput = "";
  }

  function addTag(e: KeyboardEvent | MouseEvent) {
    e.preventDefault();
    if (tag && tag !== "") {
      if (!experimentCopy.tags) {
        experimentCopy.tags = [];
      }
      experimentCopy.tags.push(tag);
      tag = null;
    }
  }
</script>

<div
  class="fixed inset-0 bg-ctp-base/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden font-mono"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-ctp-mantle w-full max-w-xl border border-ctp-surface0/30 overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <!-- HEADER -->
    <div
      class="flex items-center justify-between p-4 border-b border-ctp-surface0/20"
    >
      <div class="flex items-stretch gap-3 min-h-fit">
        <div class="w-2 bg-ctp-mauve rounded-full self-stretch"></div>
        <div class="py-1">
          <h2 id="modal-title" class="text-lg text-ctp-text font-mono">
            Edit Experiment
          </h2>
          <div class="text-sm text-ctp-subtext0">modify experiment config</div>
        </div>
      </div>
      <button
        onclick={() => {
          closeEditExperimentModal();
        }}
        type="button"
        class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 rounded p-1 transition-all"
      >
        <X size={14} />
      </button>
    </div>

    <!-- FORM -->
    <form
      method="POST"
      action="/experiments?/update"
      class="p-4 space-y-4"
      use:enhance={({ formElement, formData, action, cancel, submitter }) => {
        return async ({ result, update }) => {
          if (result.type === "success" || result.type === "redirect") {
            experiment.name = experimentCopy.name;
            experiment.description = experimentCopy.description;
            experiment.visibility = experimentCopy.visibility;
            experiment.tags = [...experiment.tags];
          }
          closeEditExperimentModal();

          await update();
        };
      }}
    >
      <input
        class="hidden"
        id="experiment-id"
        name="experiment-id"
        value={experimentCopy.id}
      />

      <div class="space-y-4">
        <!-- Basic config -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">
            experiment config
          </div>
          <div class="space-y-3">
            <div>
              <input
                id="experiment-name"
                name="experiment-name"
                type="text"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-mauve focus:border-ctp-mauve transition-all text-sm"
                placeholder="experiment_name"
                bind:value={experimentCopy.name}
                required
              />
            </div>
            <div>
              <textarea
                id="experiment-description"
                name="experiment-description"
                rows="2"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-mauve focus:border-ctp-mauve transition-all resize-none text-sm"
                placeholder="description"
                bind:value={experimentCopy.description}
                required
              ></textarea>
            </div>
          </div>
        </div>

        <!-- Visibility Setting -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">visibility</div>
          <input
            type="hidden"
            id="edit-visibility-input"
            name="visibility"
            bind:value={experimentCopy.visibility}
          />

          <div class="flex gap-2 text-sm">
            <button
              type="button"
              class={"flex items-center gap-1 px-3 py-2 transition-colors " +
                (experimentCopy.visibility === "PUBLIC"
                  ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
                  : "bg-ctp-surface0/20 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text border border-ctp-surface0/30")}
              onclick={() => (experimentCopy.visibility = "PUBLIC")}
            >
              <Globe size={12} />
              <span>public</span>
            </button>

            <button
              type="button"
              class={"flex items-center gap-1 px-3 py-2 transition-colors " +
                (experimentCopy.visibility === "PRIVATE" ||
                !experimentCopy.visibility
                  ? "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
                  : "bg-ctp-surface0/20 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text border border-ctp-surface0/30")}
              onclick={() => (experimentCopy.visibility = "PRIVATE")}
            >
              <Lock size={12} />
              <span>private</span>
            </button>
          </div>
        </div>

        <!-- Tags Section -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">tags</div>
          <div class="flex flex-wrap items-center gap-2">
            {#if experimentCopy.tags && experimentCopy.tags.length > 0}
              {#each experimentCopy.tags as tag, i}
                <input type="hidden" value={tag} name="tags.{i}" />
                <span
                  class="inline-flex items-center px-2 py-1 text-sm bg-ctp-blue/10 text-ctp-blue border border-ctp-blue/30"
                >
                  {tag}
                  <button
                    type="button"
                    class="text-ctp-blue/70 hover:text-ctp-red transition-colors ml-1.5"
                    onclick={() => experimentCopy.tags?.splice(i, 1)}
                    aria-label="Remove tag"
                  >
                    <X size={12} />
                  </button>
                </span>
              {/each}
            {/if}

            {#if addingNewTag}
              <div class="flex items-center gap-2">
                <input
                  type="text"
                  bind:value={tag}
                  class="bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 text-ctp-text focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                  placeholder="tag_name"
                  onkeydown={addTag}
                />
                <button
                  type="button"
                  onclick={addTag}
                  class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-2 py-1 text-sm transition-all"
                >
                  <Plus size={14} />
                </button>
              </div>
            {:else}
              <button
                type="button"
                onclick={(e) => {
                  e.preventDefault();
                  addingNewTag = true;
                }}
                class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-2 py-1 text-sm transition-all"
              >
                <Plus size={12} />
              </button>
            {/if}
          </div>
        </div>

        <!-- References Section -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">
            reference experiment
          </div>

          {#if reference}
            <input
              class="hidden"
              name="reference-id"
              bind:value={reference.id}
            />
            <div class="flex items-center gap-2 mb-3">
              <span class="text-ctp-lavender text-sm">•</span>
              <span class="text-ctp-text text-sm">{reference.name}</span>
              <button
                type="button"
                class="text-ctp-subtext0 hover:text-ctp-red transition-colors ml-auto"
                onclick={clearReference}
                title="Remove reference"
              >
                <X size={12} />
              </button>
            </div>
          {/if}

          <div class="space-y-2">
            <input
              type="search"
              placeholder="Search experiments..."
              bind:value={searchInput}
              class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-lavender focus:border-ctp-lavender transition-all text-sm"
            />

            <div class="max-h-32 overflow-y-auto space-y-1">
              {#each filteredExperiments as exp}
                <button
                  type="button"
                  class="w-full flex items-center gap-2 p-2 hover:bg-ctp-surface0/10 text-left text-sm transition-colors"
                  onclick={() => selectReference(exp)}
                >
                  <span class="text-ctp-lavender">•</span>
                  <span class="text-ctp-text truncate">{exp.name}</span>
                </button>
              {/each}

              {#if filteredExperiments.length === 0 && experiments.length > 1}
                <div class="p-2 text-sm text-ctp-subtext0 text-center">
                  No experiments found
                </div>
              {:else if experiments.length <= 1}
                <div class="p-2 text-sm text-ctp-subtext0 text-center">
                  No other experiments in this workspace
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>

      <!-- Footer -->
      <div
        class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      >
        <button
          onclick={() => {
            closeEditExperimentModal();
          }}
          type="button"
          class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text px-3 py-2 text-sm transition-all"
        >
          cancel
        </button>
        <button
          type="submit"
          class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-mauve hover:bg-ctp-mauve/10 hover:border-ctp-mauve/30 px-3 py-2 text-sm transition-all"
        >
          update
        </button>
      </div>
    </form>
  </div>
</div>
