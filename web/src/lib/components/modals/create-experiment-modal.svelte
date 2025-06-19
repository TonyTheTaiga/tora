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

  let {
    workspace,
    experiments,
  }: { workspace: any; experiments: Experiment[] } = $props();

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
    experiments.filter((exp) =>
      exp.name.toLowerCase().includes(searchInput.toLowerCase()),
    ),
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
  class="fixed inset-0 bg-ctp-base/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden font-mono"
>
  <div
    class="w-full max-w-xl bg-ctp-mantle border border-ctp-surface0/30 overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between p-4 border-b border-ctp-surface0/20"
    >
      <div class="flex items-stretch gap-3 min-h-fit">
        <div class="w-2 bg-ctp-mauve rounded-full self-stretch"></div>
        <div class="py-1">
          <h3 id="modal-title" class="text-lg text-ctp-text font-mono">
            New Experiment
          </h3>
          <div class="text-base text-ctp-subtext0">
            create experiment config
          </div>
        </div>
      </div>
    </div>

    <form
      method="POST"
      action="/experiments?/create"
      class="p-4 space-y-4"
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
      <div class="space-y-4">
        <!-- Basic config -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">
            experiment config
          </div>
          <div class="space-y-3">
            <div>
              <input
                name="experiment-name"
                type="text"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-mauve focus:border-ctp-mauve transition-all text-sm"
                placeholder="experiment_name"
                required
              />
            </div>
            <div>
              <textarea
                name="experiment-description"
                rows="2"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-mauve focus:border-ctp-mauve transition-all resize-none text-sm"
                placeholder="description"
                required
              ></textarea>
            </div>
          </div>
        </div>

        <!-- Visibility Setting -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">visibility</div>
          <input type="hidden" name="visibility" value={visibility} />

          <div class="flex gap-2 text-sm">
            <button
              type="button"
              class={"flex items-center gap-1 px-3 py-2 transition-colors " +
                (visibility === "PUBLIC"
                  ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
                  : "bg-ctp-surface0/20 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text border border-ctp-surface0/30")}
              onclick={() => (visibility = "PUBLIC")}
            >
              <Globe size={12} />
              <span>public</span>
            </button>

            <button
              type="button"
              class={"flex items-center gap-1 px-3 py-2 transition-colors " +
                (visibility === "PRIVATE"
                  ? "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
                  : "bg-ctp-surface0/20 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text border border-ctp-surface0/30")}
              onclick={() => (visibility = "PRIVATE")}
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
            {#each tags as tag, i}
              <input type="hidden" value={tag} name="tags.{i}" />
              <span
                class="inline-flex items-center px-2 py-1 text-sm bg-ctp-blue/10 text-ctp-blue border border-ctp-blue/30"
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
              <div class="flex items-center gap-2">
                <input
                  type="text"
                  bind:value={tag}
                  class="bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 text-ctp-text focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                  placeholder="tag_name"
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
                  class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-2 py-1 text-sm transition-all"
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
        </div>
      </div>

      <!-- Action Buttons -->
      <div
        class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      >
        <button
          onclick={() => {
            closeCreateExperimentModal();
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
          create
        </button>
      </div>
    </form>
  </div>
</div>
