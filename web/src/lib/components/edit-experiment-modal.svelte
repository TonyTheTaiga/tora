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

  let { experiment = $bindable() } = $props();

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
    user_id: experiment.user_id,
  });

  let addingNewTag = $state(false);
  let tag = $state<string | null>(null);
  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");
  const charList: string[] = [];
  let selectedIndex = $state<number>(-1);
  let searchResults = $state<Experiment[]>([]);

  onMount(async () => {
    document.body.classList.add("overflow-hidden");

    try {
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
      console.error("Failed to load reference chain:", error);
    }
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });

  async function getExperiments(query: string | null) {
    let url = `/api/experiments`;
    if (query) {
      url += `?startwith=${encodeURIComponent(query)}`;
    }

    await fetch(url, { method: "GET" })
      .then((res) => res.json())
      .then((data) => {
        searchResults = (data as Experiment[]).filter(
          (exp) => exp.id !== experiment.id,
        );
      });
  }

  async function handleKeyDown(event: KeyboardEvent) {
    const input = event.target as HTMLInputElement;

    if (searchResults.length > 0) {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, searchResults.length - 1);
        return;
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, 0);
        return;
      } else if (event.key === "Enter" && selectedIndex >= 0) {
        event.preventDefault();
        reference = searchResults[selectedIndex];
        selectedIndex = -1;
        resetSearch();
        return;
      } else if (event.key === "Escape") {
        event.preventDefault();
        searchResults = [];
        selectedIndex = -1;
        return;
      }
    }

    if (event.key === "Backspace") {
      const { selectionStart, selectionEnd } = input;

      if (selectionStart !== null && selectionEnd !== null) {
        const deleteCount =
          selectionStart !== selectionEnd ? selectionEnd - selectionStart : 1;
        const deleteIndex =
          selectionStart !== selectionEnd ? selectionStart : selectionStart - 1;

        if (deleteIndex >= 0) charList.splice(deleteIndex, deleteCount);
      } else {
        charList.pop();
      }
    } else if (/^[a-z0-9]$/i.test(event.key)) {
      charList.push(event.key);
    }

    if (charList.length && charList.length > 0) {
      await getExperiments(charList.join(""));
      selectedIndex = -1;
    } else if (charList.length === 0) {
      searchResults = [];
      selectedIndex = -1;
    }
  }

  function resetSearch() {
    searchResults = [];
    while (charList.length > 0) {
      charList.pop();
    }
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
          experiment = null;
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
          experiment = null;
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
                      onclick={() => (reference = null)}
                      aria-label="Remove reference"
                    >
                      <X size={12} />
                    </button>
                  </span>
                </div>
              {/if}

              <div class="relative">
                <input
                  id="search-input"
                  bind:value={searchInput}
                  placeholder="Search for references..."
                  class="w-full px-3 py-1.5 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-lavender transition-all placeholder-ctp-overlay0 shadow-sm"
                  onkeydown={async (event) => await handleKeyDown(event)}
                />
                {#if searchResults.length > 0}
                  <div
                    class="absolute top-full left-0 right-0 z-30 mt-1 p-2 border-0 bg-ctp-base rounded-lg shadow-xl max-h-40 overflow-y-auto"
                  >
                    <ul class="flex flex-col">
                      {#each searchResults as experiment, index}
                        <button
                          class="{selectedIndex === index
                            ? 'bg-ctp-lavender/10 text-ctp-lavender'
                            : ''} hover:bg-ctp-lavender/10 text-left px-2 py-1.5 text-xs rounded-lg text-ctp-text hover:text-ctp-lavender transition-colors"
                          onclick={(e) => {
                            e.preventDefault();
                            reference = experiment;
                            resetSearch();
                          }}
                        >
                          {experiment.name}
                        </button>
                      {/each}
                    </ul>
                  </div>
                {/if}
              </div>
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
            experiment = null;
          }}
          type="button"
          class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-lg bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve text-ctp-crust hover:shadow-lg transition-all"
        >
          <Save size={16} />
          Update
        </button>
      </div>
    </form>
  </div>
</div>
