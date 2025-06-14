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

  let { experiment, workspace, experiments }: { experiment: Experiment, workspace: any, experiments: Experiment[] } = $props();

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
    experiments.filter(exp => 
      exp.id !== experiment.id && 
      exp.name.toLowerCase().includes(searchInput.toLowerCase())
    )
  );

  onMount(async () => {
    document.body.classList.add("overflow-hidden");

    try {
      // Load existing reference
      reference = null;
      const response = await fetch(`/api/experiments/${experiment.id}/ref`);
      if (response.ok) {
        const referenceIds = await response.json();
        // Filter out self-references
        const referencesToLoad = referenceIds.filter(
          (id: String) => id !== experiment.id,
        );

        // Since we want to enforce only one reference, just use the first one
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
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-2 sm:p-4 z-50 overflow-hidden"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-ctp-mantle w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <!-- HEADER -->
    <div
      class="flex items-center justify-between px-6 py-4 border-b border-ctp-surface0"
    >
      <div class="flex items-center gap-2">
        <Save size={18} class="text-ctp-mauve" />
        <h2 id="modal-title" class="text-xl font-medium text-ctp-text">
          Edit Experiment
        </h2>
      </div>
      <button
        onclick={() => {
          closeEditExperimentModal();
        }}
        type="button"
        class="p-1.5 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/10 rounded-full transition-all"
      >
        <X size={18} />
      </button>
    </div>

    <!-- FORM -->
    <form
      method="POST"
      action="?/update"
      class="flex flex-col gap-4 p-5"
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

      <div class="flex flex-col gap-5">
        <!-- Name Input -->
        <div class="space-y-1.5">
          <label
            class="text-sm font-medium text-ctp-subtext0"
            for="experiment-name">Experiment Name</label
          >
          <input
            id="experiment-name"
            name="experiment-name"
            type="text"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
            placeholder="Enter experiment name"
            bind:value={experimentCopy.name}
            required
          />
        </div>

        <!-- Description Input -->
        <div class="space-y-1.5">
          <label
            class="text-sm font-medium text-ctp-subtext0"
            for="experiment-description"
          >
            Description
          </label>
          <textarea
            id="experiment-description"
            name="experiment-description"
            rows="2"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all resize-none placeholder-ctp-overlay0 shadow-sm"
            placeholder="Briefly describe this experiment"
            bind:value={experimentCopy.description}
            required
          ></textarea>
        </div>

        <!-- Visibility Setting -->
        <div class="space-y-1.5">
          <label
            id="edit-visibility-label"
            class="text-sm font-medium text-ctp-subtext0"
            for="visibility">Visibility</label
          >
          <input
            type="hidden"
            id="edit-visibility-input"
            name="visibility"
            bind:value={experimentCopy.visibility}
            aria-labelledby="edit-visibility-label"
          />

          <div
            class="flex gap-3"
            role="radiogroup"
            aria-labelledby="edit-visibility-label"
          >
            <button
              type="button"
              id="edit-visibility-public"
              role="radio"
              aria-checked={experimentCopy.visibility === "PUBLIC"}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (experimentCopy.visibility === "PUBLIC"
                  ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (experimentCopy.visibility = "PUBLIC")}
            >
              <Globe size={14} />
              <span>Public</span>
            </button>

            <button
              type="button"
              id="edit-visibility-private"
              role="radio"
              aria-checked={experimentCopy.visibility === "PRIVATE" ||
                !experimentCopy.visibility}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (experimentCopy.visibility === "PRIVATE" ||
                !experimentCopy.visibility
                  ? "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (experimentCopy.visibility = "PRIVATE")}
            >
              <Lock size={14} />
              <span>Private</span>
            </button>
          </div>
        </div>

        <!-- Collapsible Sections -->
        <div class="flex flex-col gap-4 mt-2">
          <!-- Tags Section -->
          <details class="group">
            <summary
              class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
            >
              <TagIcon size={16} class="text-ctp-blue" />
              <span class="text-sm font-medium">Tags</span>
              <ChevronDown
                size={16}
                class="ml-auto text-ctp-subtext0 group-open:rotate-180"
              />
            </summary>
            <div class="pt-2 pl-6">
              <div class="flex flex-wrap items-center gap-2">
                {#if experimentCopy.tags && experimentCopy.tags.length > 0}
                  {#each experimentCopy.tags as tag, i}
                    <input type="hidden" value={tag} name="tags.{i}" />
                    <span
                      class="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full bg-ctp-blue/10 text-ctp-blue border-0 group"
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
                  <div class="flex items-center gap-1">
                    <input
                      type="text"
                      bind:value={tag}
                      class="w-40 px-2 py-1.5 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
                      placeholder="New tag"
                      onkeydown={addTag}
                    />
                    <button
                      type="button"
                      onclick={addTag}
                      class="p-1.5 rounded-full text-ctp-blue hover:bg-ctp-blue/10 transition-all"
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
                    class="inline-flex items-center gap-1 py-0.5 px-2 text-xs rounded-full bg-transparent text-ctp-blue border border-dashed border-ctp-blue/50 hover:bg-ctp-blue/10 transition-all"
                  >
                    <Plus size={12} />
                    Add Tag
                  </button>
                {/if}
              </div>
            </div>
          </details>

          <!-- References Section -->
          <details class="group">
            <summary
              class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
            >
              <Link size={16} class="text-ctp-lavender" />
              <span class="text-sm font-medium">References</span>
              <ChevronDown
                size={16}
                class="ml-auto text-ctp-subtext0 group-open:rotate-180"
              />
            </summary>
            <div class="pt-2 pl-6">
              {#if reference}
                <input
                  class="hidden"
                  name="reference-id"
                  bind:value={reference.id}
                />
                <div class="mb-3">
                  <span
                    class="inline-flex items-center px-2 py-1 text-xs rounded-lg bg-ctp-lavender/10 text-ctp-lavender border-0"
                  >
                    <span title="Referenced experiment">{reference.name}</span>
                    <button
                      type="button"
                      class="text-ctp-lavender/70 hover:text-ctp-red transition-colors ml-1.5"
                      onclick={clearReference}
                      aria-label="Remove reference"
                    >
                      <X size={12} />
                    </button>
                  </span>
                </div>
              {/if}

              <!-- Dropdown selector -->
              <details class="relative">
                <summary
                  class="flex items-center justify-between cursor-pointer p-2 hover:bg-ctp-surface1 transition-colors rounded-lg"
                >
                  <span class="text-sm text-ctp-text">
                    Select reference experiment
                  </span>
                  <ChevronDown size={16} class="text-ctp-subtext1" />
                </summary>

                <div
                  class="absolute top-full left-0 right-0 mt-1 z-30 max-h-60 overflow-y-auto border border-ctp-surface1/30 bg-ctp-surface0/80 backdrop-blur-sm rounded-lg shadow-lg"
                >
                  <!-- Search filter -->
                  <div class="p-2 border-b border-ctp-surface1/20">
                    <input
                      type="search"
                      placeholder="Filter experiments..."
                      bind:value={searchInput}
                      class="w-full px-2 py-1 text-sm bg-ctp-base border border-ctp-surface1 rounded text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:border-ctp-lavender"
                    />
                  </div>

                  <!-- Control buttons -->
                  <div class="flex gap-2 p-2 border-b border-ctp-surface1/20">
                    <button
                      onclick={clearReference}
                      type="button"
                      class="px-2 py-1 text-xs bg-ctp-red/20 text-ctp-red rounded hover:bg-ctp-red/30 transition-colors"
                    >
                      Clear Selection
                    </button>
                    <button
                      onclick={clearSearch}
                      type="button"
                      class="px-2 py-1 text-xs bg-ctp-blue/20 text-ctp-blue rounded hover:bg-ctp-blue/30 transition-colors"
                    >
                      Clear Filter
                    </button>
                  </div>

                  <!-- Experiment list -->
                  <div class="p-1">
                    {#each filteredExperiments as exp}
                      <button
                        type="button"
                        class="w-full flex items-center gap-2 p-2 hover:bg-ctp-surface1 rounded cursor-pointer text-left"
                        onclick={() => selectReference(exp)}
                      >
                        <div class="w-2 h-2 rounded-full bg-ctp-lavender flex-shrink-0"></div>
                        <span class="text-sm text-ctp-text truncate">{exp.name}</span>
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
              </details>
            </div>
          </details>
        </div>
      </div>

      <!-- Footer -->
      <div
        class="flex justify-end gap-3 pt-4 mt-2 border-t border-ctp-surface0"
      >
        <button
          onclick={() => {
            closeEditExperimentModal();
          }}
          type="button"
          class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-full bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-full bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all"
        >
          <Save size={16} />
          Update
        </button>
      </div>
    </form>
  </div>
</div>
