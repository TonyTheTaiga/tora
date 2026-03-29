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
  import { Button } from "$lib/components";

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
              class="inline-flex items-center gap-1.5 text-[11px] font-mono text-ctp-subtext1"
            >
              <span class="text-ctp-blue/60">#</span>{tag}
              <button
                type="button"
                class="text-ctp-overlay0 hover:text-ctp-red transition-colors"
                onclick={() => tags.splice(i, 1)}
                aria-label="Remove tag"
              >
                <X size={11} />
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
              <Button
                type="button"
                variant="primary"
                onclick={(event) => {
                  event.preventDefault();
                  addTag();
                }}
              >
                <Plus size={14} />
              </Button>
            </div>
          {:else}
            <Button
              type="button"
              variant="primary"
              onclick={(event) => {
                event.preventDefault();
                addingNewTag = true;
              }}
            >
              <Plus size={14} />
            </Button>
          {/if}
        </div>
      </ModalFormSection>
    </div>

    <ModalButtons onCancel={closeCreateExperimentModal} submitText="create" />
  </form>
</BaseModal>
