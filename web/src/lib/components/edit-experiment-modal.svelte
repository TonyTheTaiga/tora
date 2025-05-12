<script lang="ts">
  import { X, Save, TagIcon, Plus, Link } from "lucide-svelte";
  import { enhance } from "$app/forms";
  import type { Experiment } from "$lib/types";

  let { experiment = $bindable(), editMode = $bindable() } = $props();

  let addingNewTag = $state(false);
  let tag = $state<string | null>(null);
  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");
  const charList: string[] = [];
  let selectedIndex = $state<number>(-1);
  let searchResults = $state<Experiment[]>([]);

  async function getExperiments(query: string | null) {
    let url = `/api/experiments`;
    if (query) {
      url += `?startwith=${encodeURIComponent(query)}`;
    }

    await fetch(url, { method: "GET" })
      .then((res) => res.json())
      .then((data) => {
        searchResults = data as Experiment[];
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
      experiment.tags.push(tag);
      tag = null;
    }
  }
</script>

<div
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-4 z-50"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-ctp-mantle w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-hidden"
  >
    <!-- HEADER -->
    <div
      class="px-6 py-4 border-b border-ctp-surface0 flex justify-between items-center"
    >
      <h2 class="text-xl font-medium text-ctp-text flex items-center gap-2">
        <Save size={18} class="text-ctp-mauve" />
        Edit Experiment
      </h2>
      <button
        onclick={() => {
          editMode = !editMode;
        }}
        class="p-2 text-ctp-subtext0 hover:text-ctp-text hover:bg-ctp-surface0/50 rounded-full transition-all"
        aria-label="Close modal"
      >
        <X size={18} />
      </button>
    </div>

    <!-- FORM -->
    <div class="p-6">
      <form
        method="POST"
        action="?/update"
        class="flex flex-col gap-8"
        use:enhance={({ formElement, formData, action, cancel, submitter }) => {
          experiment.name = formData.get("experiment-name");
          experiment.description = formData.get("experiment-description");
          return async ({ result, update }) => {
            editMode = !editMode;
          };
        }}
      >
        <input
          class="hidden"
          id="experiment-id"
          name="experiment-id"
          value={experiment.id}
        />

        <!-- Basic Info Section -->
        <div class="space-y-5">
          <!-- Name Input -->
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-ctp-subtext0"
              for="experiment-name">Experiment Name</label
            >
            <input
              id="experiment-name"
              name="experiment-name"
              type="text"
              class="w-full px-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
              placeholder="Enter experiment name"
              value={experiment.name}
              required
            />
          </div>

          <!-- Description Input -->
          <div class="space-y-2">
            <label
              class="text-sm font-medium text-ctp-subtext0"
              for="experiment-description"
            >
              Description
            </label>
            <textarea
              id="experiment-description"
              name="experiment-description"
              rows="3"
              class="w-full px-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all resize-none placeholder-ctp-overlay0 shadow-sm"
              placeholder="Briefly describe this experiment"
              value={experiment.description}
              required
            ></textarea>
          </div>
        </div>

        <!-- Tags Section -->
        <div class="space-y-4">
          <div
            class="flex items-center gap-3 pb-2 border-b border-ctp-surface0"
          >
            <TagIcon size={18} class="text-ctp-pink" />
            <h3 class="text-xl font-medium text-ctp-text">Tags</h3>
          </div>

          <div class="flex flex-wrap items-center gap-2">
            {#each experiment.tags as tag, i}
              <input type="hidden" value={tag} name="tags.{i}" />
              <span
                class="inline-flex items-center px-3 py-1 text-xs font-medium rounded-full bg-ctp-mauve/10 text-ctp-mauve border-0 group"
              >
                {tag}
                <button
                  type="button"
                  class="text-ctp-mauve/70 hover:text-ctp-red transition-colors ml-2"
                  onclick={() => experiment.tags.splice(i, 1)}
                  aria-label="Remove tag"
                >
                  <X size={14} />
                </button>
              </span>
            {/each}

            {#if addingNewTag}
              <div class="flex items-center gap-1">
                <input
                  type="text"
                  bind:value={tag}
                  class="w-40 px-3 py-2 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
                  placeholder="New tag"
                  onkeydown={addTag}
                />
                <button
                  type="button"
                  onclick={addTag}
                  class="p-2 rounded-full text-ctp-mauve hover:bg-ctp-mauve/10 transition-all"
                >
                  <Plus size={16} />
                </button>
              </div>
            {:else}
              <button
                type="button"
                onclick={(e) => {
                  e.preventDefault();
                  addingNewTag = true;
                }}
                class="inline-flex items-center gap-1 py-1 px-3 text-sm rounded-full bg-transparent text-ctp-mauve border border-dashed border-ctp-mauve/50 hover:bg-ctp-mauve/10 transition-all"
              >
                <Plus size={14} />
                Add Tag
              </button>
            {/if}
          </div>
        </div>

        <!-- Reference -->
        <div class="space-y-4">
          {#if reference}
            <input
              class="hidden"
              name="reference-id"
              bind:value={reference.id}
            />
          {/if}

          <div
            class="flex items-center gap-3 pb-2 border-b border-ctp-surface0"
          >
            <Link size={18} class="text-ctp-lavender" />
            <h3 class="text-xl font-medium text-ctp-text">References</h3>
          </div>

          <div>
            {#if reference}
              <span
                class="inline-flex items-center px-3 py-1.5 text-sm rounded-lg bg-ctp-lavender/10 text-ctp-lavender border-0"
              >
                {reference.name}
                <button
                  type="button"
                  class="text-ctp-lavender/70 hover:text-ctp-red transition-colors ml-2"
                  onclick={() => (reference = null)}
                  aria-label="Remove reference"
                >
                  <X size={14} />
                </button>
              </span>
            {/if}
          </div>

          <div class="flex flex-col space-y-2 relative">
            <div>
              <input
                id="search-input"
                bind:value={searchInput}
                placeholder="Search for references..."
                class="w-full px-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-lavender transition-all placeholder-ctp-overlay0 shadow-sm"
                onkeydown={async (event) => await handleKeyDown(event)}
              />
            </div>
            {#if searchResults.length > 0}
              <div
                class="absolute top-full left-0 right-0 z-10 mt-1 p-2 border-0 bg-ctp-base rounded-lg shadow-xl max-h-60 overflow-y-auto"
              >
                <ul class="flex flex-col">
                  {#each searchResults as experiment, index}
                    <button
                      class="{selectedIndex === index
                        ? 'bg-ctp-lavender/10 text-ctp-lavender'
                        : ''} hover:bg-ctp-lavender/10 text-left px-3 py-2 rounded-lg text-ctp-text hover:text-ctp-lavender transition-colors"
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

        <!-- Footer -->
        <div
          class="flex justify-end gap-3 pt-6 mt-2 border-t border-ctp-surface0"
        >
          <button
            onclick={() => {
              editMode = !editMode;
            }}
            type="button"
            class="inline-flex items-center justify-center px-5 py-2.5 font-medium rounded-lg bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            class="inline-flex items-center justify-center gap-2 px-5 py-2.5 font-medium rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve text-ctp-crust hover:shadow-lg transition-all"
          >
            <Save size={18} />
            Update Experiment
          </button>
        </div>
      </form>
    </div>
    <!-- END FORM -->
  </div>
</div>
