<script lang="ts">
  import { Plus, X } from "@lucide/svelte";
  import { closeCreateExperimentModal } from "$lib/state/modal.svelte.js";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

  let { workspace }: { workspace?: any } = $props();

  let experimentName = $state("");
  let experimentDescription = $state("");
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
</script>

<BaseModal title="New Experiment">
  <form
    method="POST"
    action="/experiments?/create"
    class="space-y-4"
    use:enhance={() => {
      return async ({ result, update }) => {
        await update();
        if (result.type === "redirect") {
          goto(result.location);
        } else if (result.type === "success") {
          closeCreateExperimentModal();
        }
      };
    }}
  >
    <div class="space-y-4">
      <ModalFormSection title="experiment config">
        <div>
          <ModalInput
            name="experiment-name"
            placeholder="experiment_name"
            bind:value={experimentName}
            required
          />
        </div>
        <div>
          <ModalInput
            name="experiment-description"
            type="textarea"
            rows={2}
            placeholder="description"
            bind:value={experimentDescription}
            required
          />
        </div>
      </ModalFormSection>

      <ModalFormSection title="tags">
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
                class="bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
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
                class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-3 py-2 text-sm transition-all"
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
              class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-3 py-2 text-sm transition-all"
            >
              <Plus size={14} />
            </button>
          {/if}
        </div>
      </ModalFormSection>
    </div>

    <ModalButtons onCancel={closeCreateExperimentModal} submitText="create" />
  </form>
</BaseModal>
