<script lang="ts">
  import { X, Save, TagIcon, Plus } from "lucide-svelte";
  import { enhance } from "$app/forms";
  import type { Experiment } from "$lib/types";
  import { onMount, onDestroy } from "svelte";
  import { closeEditExperimentModal } from "$lib/state/app.svelte.js";

  let {
    experiment = $bindable(),
  }: {
    experiment: Experiment;
  } = $props();

  let experimentCopy = $state<Experiment>({
    id: experiment.id,
    name: experiment.name,
    description: experiment.description,
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

  onMount(async () => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });

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
      class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
    >
      <div
        class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
      >
        <div
          class="w-2 bg-ctp-mauve rounded-full flex-shrink-0 self-stretch"
        ></div>
        <div class="min-w-0 flex-1 py-1">
          <h2
            id="modal-title"
            class="text-lg md:text-xl text-ctp-text truncate font-mono"
          >
            Edit Experiment
          </h2>
          <div class="text-sm text-ctp-subtext0 space-y-1">
            <div>modify experiment config</div>
          </div>
        </div>
      </div>
      <button
        onclick={() => {
          closeEditExperimentModal();
        }}
        type="button"
        class="text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-surface0/30 rounded p-1 transition-all flex-shrink-0"
      >
        <X size={14} />
      </button>
    </div>

    <!-- FORM -->
    <form
      method="POST"
      action="/experiments?/update"
      class="px-4 md:px-6 py-4 space-y-4"
      use:enhance={({ formElement, formData, action, cancel, submitter }) => {
        return async ({ result, update }) => {
          if (result.type === "success" || result.type === "redirect") {
            experiment.name = experimentCopy.name;
            experiment.description = experimentCopy.description;
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
