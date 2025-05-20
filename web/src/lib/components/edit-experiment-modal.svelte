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
  import type { Visibility } from "$lib/types";
  import { enhance } from "$app/forms";
  import type { Experiment } from "$lib/types";
  import { onMount, onDestroy } from "svelte";

  let { experiment = $bindable(), editMode = $bindable() } = $props();
  let addingNewTag = $state(false);
  let tag = $state<string | null>(null);
  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");
  const charList: string[] = [];
  let selectedIndex = $state<number>(-1);
  let searchResults = $state<Experiment[]>([]);

  onMount(() => {
    document.body.classList.add("overflow-hidden");
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
         flex items-center justify-center p-2 sm:p-4 z-50 overflow-hidden"
>
  <!-- MODAL CONTAINER -->
  <div
    class="bg-ctp-mantle w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-auto max-h-[90vh]"
  >
    <!-- HEADER -->
    <div
      class="flex items-center justify-between px-6 py-4 border-b border-ctp-surface0"
    >
      <div class="flex items-center gap-2">
        <Save size={18} class="text-ctp-mauve" />
        <h2 class="text-xl font-medium text-ctp-text">Edit Experiment</h2>
      </div>
      <button
        onclick={() => (editMode = false)}
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
        experiment.name = formData.get("experiment-name");
        experiment.description = formData.get("experiment-description");
        experiment.visibility = formData.get("visibility") as Visibility;
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
            value={experiment.name}
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
            value={experiment.description}
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
            value={experiment.visibility}
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
              aria-checked={experiment.visibility === "PUBLIC"}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (experiment.visibility === "PUBLIC"
                  ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (experiment.visibility = "PUBLIC")}
            >
              <Globe size={14} />
              <span>Public</span>
            </button>

            <button
              type="button"
              id="edit-visibility-private"
              role="radio"
              aria-checked={experiment.visibility === "PRIVATE" ||
                !experiment.visibility}
              class={"flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors " +
                (experiment.visibility === "PRIVATE" || !experiment.visibility
                  ? "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
                  : "bg-ctp-surface0/50 text-ctp-subtext0 hover:bg-ctp-surface0 hover:text-ctp-text")}
              onclick={() => (experiment.visibility = "PRIVATE")}
            >
              <Lock size={14} />
              <span>Private</span>
            </button>
          </div>
        </div>

        <!-- Collapsible Sections -->
        <div class="flex flex-col gap-4 mt-2">
          <!-- Tags Section -->
          <details>
            <summary
              class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
            >
              <TagIcon size={16} class="text-ctp-blue" />
              <span class="text-sm font-medium">Tags</span>
              <ChevronDown size={16} class="ml-auto text-ctp-subtext0" />
            </summary>
            <div class="pt-2 pl-6">
              <div class="flex flex-wrap items-center gap-2">
                {#each experiment.tags as tag, i}
                  <input type="hidden" value={tag} name="tags.{i}" />
                  <span
                    class="inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full bg-ctp-blue/10 text-ctp-blue border-0 group"
                  >
                    {tag}
                    <button
                      type="button"
                      class="text-ctp-blue/70 hover:text-ctp-red transition-colors ml-1.5"
                      onclick={() => experiment.tags.splice(i, 1)}
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
          <details>
            <summary
              class="flex items-center gap-2 cursor-pointer text-ctp-subtext0 hover:text-ctp-text py-1.5"
            >
              <Link size={16} class="text-ctp-lavender" />
              <span class="text-sm font-medium">References</span>
              <ChevronDown size={16} class="ml-auto text-ctp-subtext0" />
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
                    {reference.name}
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
                    class="absolute top-full left-0 right-0 z-10 mt-1 p-2 border-0 bg-ctp-base rounded-lg shadow-xl max-h-40 overflow-y-auto"
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
            editMode = !editMode;
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
