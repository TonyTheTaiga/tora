<script lang="ts">
  import { X, Plus } from "@lucide/svelte";
  import { enhance } from "$app/forms";
  import type { Experiment } from "$lib/types";
  import { closeEditExperimentModal } from "$lib/state/app.svelte.js";
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

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

  function addTag() {
    if (tag && tag.trim() !== "") {
      if (!experimentCopy.tags) {
        experimentCopy.tags = [];
      }
      experimentCopy.tags.push(tag.trim());
      tag = null;
      addingNewTag = false;
    }
  }
</script>

<BaseModal title="Edit Experiment">
  {#snippet children()}
    <form
      method="POST"
      action="/experiments?/update"
      class="space-y-4"
      use:enhance={() => {
        return async ({ result, update }) => {
          await update();
          if (result.type === "success" || result.type === "redirect") {
            experiment.name = experimentCopy.name;
            experiment.description = experimentCopy.description;
            experiment.tags = [...experiment.tags];
            closeEditExperimentModal();
          }
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
        <ModalFormSection title="experiment config">
          {#snippet children()}
            <div>
              <ModalInput
                id="experiment-name"
                name="experiment-name"
                placeholder="experiment_name"
                bind:value={experimentCopy.name}
                required
              />
            </div>
            <div>
              <ModalInput
                id="experiment-description"
                name="experiment-description"
                type="textarea"
                rows={2}
                placeholder="description"
                bind:value={experimentCopy.description}
                required
              />
            </div>
          {/snippet}
        </ModalFormSection>

        <ModalFormSection title="tags">
          {#snippet children()}
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
          {/snippet}
        </ModalFormSection>
      </div>

      <ModalButtons onCancel={closeEditExperimentModal} submitText="update" />
    </form>
  {/snippet}
</BaseModal>
