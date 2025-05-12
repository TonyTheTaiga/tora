<script lang="ts">
  import type { HyperParam, Experiment } from "../types";
  import {
    Plus,
    X,
    Tag as TagIcon,
    Settings,
    Beaker,
    Link,
  } from "lucide-svelte";

  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();

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

  let searchResults = $state<Experiment[]>([]);
  let reference = $state<Experiment | null>(null);
  let searchInput = $state<string>("");
  const charList: string[] = [];
  let selectedIndex = $state<number>(-1);

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
</script>

<form
  method="POST"
  action="?/create"
  class="flex flex-col gap-8 p-6 bg-ctp-mantle rounded-xl border border-ctp-surface0 shadow-lg"
>
  <div class="flex flex-col gap-8">
    <!-- Basic Information Section -->
    <div class="space-y-5">
      <div class="flex items-center gap-3 pb-2 border-b border-ctp-surface0">
        <Beaker size={18} class="text-ctp-mauve" />
        <h3 class="text-xl font-medium text-ctp-text">Basic Information</h3>
      </div>

      <!-- Name Input -->
      <div class="space-y-2">
        <label
          class="text-sm font-medium text-ctp-subtext0"
          for="experiment-name"
        >
          Experiment Name
        </label>
        <input
          name="experiment-name"
          type="text"
          class="w-full px-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
          placeholder="Enter experiment name"
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
          name="experiment-description"
          rows="3"
          class="w-full px-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all resize-none placeholder-ctp-overlay0 shadow-sm"
          placeholder="Briefly describe this experiment"
          required
        ></textarea>
      </div>
    </div>

    <!-- Tags Section -->
    <div class="space-y-4">
      <div class="flex items-center gap-3 pb-2 border-b border-ctp-surface0">
        <TagIcon size={18} class="text-ctp-pink" />
        <h3 class="text-xl font-medium text-ctp-text">Tags</h3>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        {#each tags as tag, i}
          <input type="hidden" value={tag} name="tags.{i}" />
          <span
            class="inline-flex items-center px-3 py-1 text-xs font-medium rounded-full bg-ctp-pink/10 text-ctp-pink border-0 group"
          >
            {tag}
            <button
              type="button"
              class="text-ctp-pink/70 hover:text-ctp-red transition-colors ml-2"
              onclick={() => tags.splice(i, 1)}
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
              class="w-40 px-3 py-2 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-pink transition-all placeholder-ctp-overlay0 shadow-sm"
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
              class="p-2 rounded-full text-ctp-pink hover:bg-ctp-pink/10 transition-all"
            >
              <Plus size={16} />
            </button>
          </div>
        {:else}
          <button
            type="button"
            onclick={(event) => {
              event.preventDefault();
              addingNewTag = true;
            }}
            class="inline-flex items-center gap-1 py-1 px-3 text-sm rounded-full bg-transparent text-ctp-pink border border-dashed border-ctp-pink/50 hover:bg-ctp-pink/10 transition-all"
          >
            <Plus size={14} />
            Add Tag
          </button>
        {/if}
      </div>
    </div>

    <!-- Hyperparameters Section -->
    <div class="space-y-4">
      <div class="flex items-center gap-3 pb-2 border-b border-ctp-surface0">
        <Settings size={18} class="text-ctp-sapphire" />
        <h3 class="text-xl font-medium text-ctp-text">Parameters</h3>
      </div>

      <div class="space-y-3">
        {#each hyperparams as pair, i}
          <div class="flex gap-3 items-center">
            <input
              class="w-full px-4 py-3 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-sapphire transition-all flex-1 placeholder-ctp-overlay0 shadow-sm"
              name="hyperparams.{i}.key"
              placeholder="Parameter name"
              required
            />
            <input
              class="w-full px-4 py-3 text-sm bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-sapphire transition-all flex-1 placeholder-ctp-overlay0 shadow-sm"
              name="hyperparams.{i}.value"
              placeholder="Value"
              required
            />
            <button
              type="button"
              class="p-2 text-ctp-subtext0 hover:text-ctp-red hover:bg-ctp-red/10 rounded-full transition-all"
              onclick={() => hyperparams.splice(i, 1)}
            >
              <X size={18} />
            </button>
          </div>
        {/each}

        <button
          type="button"
          class="inline-flex items-center gap-2 py-2 px-4 text-sm font-medium rounded-lg bg-ctp-sapphire/10 text-ctp-sapphire border border-dashed border-ctp-sapphire/50 hover:bg-ctp-sapphire/20 transition-all"
          onclick={() =>
            (hyperparams = [...hyperparams, { key: "", value: "" }])}
        >
          <Plus size={16} />
          Add Parameter
        </button>
      </div>
    </div>

    <!-- References -->
    <div class="space-y-4">
      {#if reference}
        <input class="hidden" name="reference-id" bind:value={reference.id} />
      {/if}

      <div class="flex items-center gap-3 pb-2 border-b border-ctp-surface0">
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
  </div>

  <!-- Action Buttons -->
  <div class="flex justify-end gap-3 pt-6 mt-2 border-t border-ctp-surface0">
    <button
      onclick={toggleIsOpen}
      type="button"
      class="inline-flex items-center justify-center px-5 py-2.5 font-medium rounded-lg bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
    >
      Cancel
    </button>
    <button
      type="submit"
      class="inline-flex items-center justify-center gap-2 px-5 py-2.5 font-medium rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve text-ctp-crust hover:shadow-lg transition-all"
    >
      <Plus size={18} />
      Create Experiment
    </button>
  </div>
</form>
