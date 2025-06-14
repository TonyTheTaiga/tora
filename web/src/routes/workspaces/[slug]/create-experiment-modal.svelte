<script lang="ts">
  import type { HyperParam, Experiment, Visibility } from "$lib/types";
  import {
    Plus,
    X,
    Tag as TagIcon,
    Settings,
    Beaker,
    Link,
    Globe,
    Lock,
    ChevronDown,
  } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";
  import { closeCreateExperimentModal } from "$lib/state/app.svelte.js";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";

  let { workspace, experiments }: { workspace: any, experiments: Experiment[] } = $props();

  let hyperparams = $state<HyperParam[]>([]);
  let addingNewTag = $state<boolean>(false);
  let tag = $state<string | null>(null);
  let tags = $state<string[]>([]);
  let visibility = $state<Visibility>("PRIVATE");

  function addTag() {
    if (tag) {
      tags = [...tags, tag];
      tag = null;
      addingNewTag = false;
    }
  }

  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");
  
  let filteredExperiments = $derived(
    experiments.filter(exp => 
      exp.name.toLowerCase().includes(searchInput.toLowerCase())
    )
  );

  function selectReference(exp: Experiment) {
    reference = exp;
  }

  function clearReference() {
    reference = null;
  }

  function clearSearch() {
    searchInput = "";
  }

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });
</script>

<div
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-2 sm:p-4 z-50 overflow-hidden"
>
  <div
    class="w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-auto max-h-[90vh] bg-ctp-mantle"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between px-6 py-4 border-b border-ctp-surface0"
    >
      <div class="flex items-center gap-2">
        <Beaker size={18} class="text-ctp-mauve" />
        <h3 id="modal-title" class="text-xl font-medium text-ctp-text">
          New Experiment
        </h3>
      </div>
    </div>

    <form
      method="POST"
      action="?/create"
      class="flex flex-col gap-4 p-5"
      use:enhance={({ formElement, formData, action, cancel }) => {
        formData.append("workspace-id", workspace.id);
        return async ({ result, update }) => {
          if (result.type === "redirect") {
            goto(result.location);
          } else {
            await update();
            closeCreateExperimentModal();
          }
        };
      }}
    >
      <div class="flex flex-col gap-5">
        <!-- Name Input -->
        <div class="space-y-1.5">
          <label
            class="text-sm font-medium text-ctp-subtext0"
            for="experiment-name"
          >
            Experiment Name
          </label>
          <input
            name="experiment-name"
            type="text"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
            placeholder="Enter experiment name"
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
            name="experiment-description"
            rows="2"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all resize-none placeholder-ctp-overlay0 shadow-sm"
            placeholder="Briefly describe this experiment"
            required
          ></textarea>
        </div>

        <!-- Visibility Setting -->
        <div class="space-y-1.5">
          <label
            id="create-visibility-label"
            class="text-sm font-medium text-ctp-subtext0"
            for="visibility">Visibility</label
          >
          <input
            type="hidden"
            id="create-visibility-input"
            name="visibility"
            value={visibility}
            aria-labelledby="create-visibility-label"
          />

          <div
            class="flex gap-3"
            role="radiogroup"
            aria-labelledby="create-visibility-label"
          >
            <button
              type="button"
              id="create-visibility-public"
              role="radio"
              aria-checked={visibility === "PUBLIC"}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (visibility === "PUBLIC"
                  ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (visibility = "PUBLIC")}
            >
              <Globe size={14} />
              <span>Public</span>
            </button>

            <button
              type="button"
              id="create-visibility-private"
              role="radio"
              aria-checked={visibility === "PRIVATE"}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (visibility === "PRIVATE"
                  ? "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (visibility = "PRIVATE")}
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
                {#each tags as tag, i}
                  <input type="hidden" value={tag} name="tags.{i}" />
                  <span
                    class="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full bg-ctp-blue/10 text-ctp-blue border-0 group"
                  >
                    {tag}
                    <button
                      type="button"
                      class="text-ctp-blue/70 hover:text-ctp-red transition-colors ml-1.5"
                      onclick={() => tags.splice(i, 1)}
                      aria-label="Remove tag"
                    >
                      <X size={12} />
                    </button>
                  </span>
                {/each}

                {#if addingNewTag}
                  <div class="flex items-center gap-1">
                    <input
                      type="text"
                      bind:value={tag}
                      class="w-40 px-2 py-1.5 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
                      placeholder="New tag"
                      onkeydown={(event) => {
                        if (event.key === "Enter") {
                          event.preventDefault();
                          addTag();
                        }
                      }}
                    />
                    <button
                      type="button"
                      onclick={(event) => {
                        event.preventDefault();
                        addTag();
                      }}
                      class="p-1.5 rounded-full text-ctp-blue hover:bg-ctp-blue/10 transition-all"
                    >
                      <Plus size={14} />
                    </button>
                  </div>
                {:else}
                  <button
                    type="button"
                    onclick={(event) => {
                      event.preventDefault();
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

          <!-- Parameters Section -->
          <details class="group">
            <summary
              class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
            >
              <Settings size={16} class="text-ctp-sapphire" />
              <span class="text-sm font-medium">Parameters</span>
              <ChevronDown
                size={16}
                class="ml-auto text-ctp-subtext0 group-open:rotate180"
              />
            </summary>
            <div class="pt-2 pl-6">
              <div class="space-y-3">
                {#each hyperparams as pair, i}
                  <div class="flex gap-2 items-center">
                    <input
                      class="w-full px-3 py-1.5 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-sapphire transition-all flex-1 placeholder-ctp-overlay0 shadow-sm"
                      name="hyperparams.{i}.key"
                      placeholder="Parameter name"
                      required
                    />
                    <input
                      class="w-full px-3 py-1.5 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-sapphire transition-all flex-1 placeholder-ctp-overlay0 shadow-sm"
                      name="hyperparams.{i}.value"
                      placeholder="Value"
                      required
                    />
                    <button
                      type="button"
                      class="p-1.5 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/10 rounded-full transition-all"
                      onclick={() => hyperparams.splice(i, 1)}
                    >
                      <X size={16} />
                    </button>
                  </div>
                {/each}

                <button
                  type="button"
                  class="inline-flex items-center gap-1.5 py-1.5 px-3 text-sm font-medium rounded-lg bg-ctp-sapphire/10 text-ctp-sapphire border border-dashed border-ctp-sapphire/50 hover:bg-ctp-sapphire/20 transition-all"
                  onclick={() =>
                    (hyperparams = [...hyperparams, { key: "", value: "" }])}
                >
                  <Plus size={14} />
                  Add Parameter
                </button>
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

                    {#if filteredExperiments.length === 0 && experiments.length > 0}
                      <div class="p-2 text-sm text-ctp-subtext0 text-center">
                        No experiments found
                      </div>
                    {:else if experiments.length === 0}
                      <div class="p-2 text-sm text-ctp-subtext0 text-center">
                        No experiments in this workspace yet
                      </div>
                    {/if}
                  </div>
                </div>
              </details>
            </div>
          </details>
        </div>
      </div>

      <!-- Action Buttons -->
      <div
        class="flex justify-end gap-3 pt-4 mt-2 border-t border-ctp-surface0"
      >
        <button
          onclick={() => {
            closeCreateExperimentModal();
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
          <Plus size={16} />
          Create
        </button>
      </div>
    </form>
  </div>
</div>
