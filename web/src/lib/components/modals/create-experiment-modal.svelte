<script lang="ts">
  import type { HyperParam, Experiment } from "$lib/types";
  import {
    Plus,
    X,
    Tag as TagIcon,
    Settings,
    Beaker,
  } from "lucide-svelte";
  import { onMount, onDestroy } from "svelte";
  import { closeCreateExperimentModal } from "$lib/state/app.svelte.js";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";

  let {
    workspace,
  }: { workspace: any } = $props();

  let hyperparams = $state<HyperParam[]>([]);
  let addingNewTag = $state<boolean>(false);
  let tag = $state<string | null>(null);
  let tags = $state<string[]>([]);

  function addTag() {
    if (tag) {
      tags = [...tags, tag];
      tag = null;
      addingNewTag = false;
    }
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
      class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/10"
    >
      <div
        class="flex items-stretch gap-3 md:gap-4 min-w-0 flex-1 pr-4 min-h-fit"
      >
        <div
          class="w-2 bg-ctp-mauve rounded-full flex-shrink-0 self-stretch"
        ></div>
        <div class="min-w-0 flex-1 py-1">
          <h3
            id="modal-title"
            class="text-lg md:text-xl text-ctp-text truncate font-mono"
          >
            New Experiment
          </h3>
          <div class="text-sm text-ctp-subtext0 space-y-1">
            <div>create experiment config</div>
          </div>
        </div>
      </div>
    </div>

    <form
      method="POST"
      action="/experiments?/create"
      class="px-4 md:px-6 py-4 space-y-4"
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
