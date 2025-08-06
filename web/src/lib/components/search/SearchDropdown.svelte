<script lang="ts" generics="T">
  import { onMount } from "svelte";
  import { ChevronDown } from "@lucide/svelte";
  import {
    createSearchKeyboardHandler,
    toggleArrayItem,
    isItemSelected,
  } from "./search-utils.js";

  interface Props<T> {
    items: T[];
    selectedItems?: T[];
    searchQuery?: string;
    placeholder?: string;
    summaryText?: string;
    itemTypeName?: string;
    showControlButtons?: boolean;
    enableKeyboardShortcuts?: boolean;
    enableEscapeToBlur?: boolean;
    getItemText: (item: T) => string;
    getItemKey?: (item: T) => string | number;
    onSelectionChange?: (selectedItems: T[]) => void;
    onSearchChange?: (query: string) => void;
  }

  let {
    items,
    selectedItems = $bindable([]),
    searchQuery = $bindable(""),
    placeholder = "filter...",
    summaryText,
    itemTypeName = "items",
    showControlButtons = true,
    enableKeyboardShortcuts = true,
    enableEscapeToBlur = true,
    getItemText,
    getItemKey = (item: T) => getItemText(item),
    onSelectionChange,
    onSearchChange,
  }: Props<T> = $props();

  let detailsElement: HTMLDetailsElement | null = $state(null);
  let searchInputElement: HTMLInputElement | null = $state(null);

  const filteredItems = $derived.by(() => {
    if (!searchQuery.trim()) return items;
    const query = searchQuery.toLowerCase().trim();
    return items.filter((item) =>
      getItemText(item).toLowerCase().includes(query),
    );
  });

  const summaryDisplayText = $derived(() => {
    if (summaryText) return summaryText;
    return `select ${itemTypeName} (${selectedItems.length}/${items.length})`;
  });

  function selectAllItems() {
    selectedItems = [...items];
    onSelectionChange?.(selectedItems);
  }

  function clearAllItems() {
    selectedItems = [];
    onSelectionChange?.(selectedItems);
  }

  function toggleItem(item: T) {
    selectedItems = toggleArrayItem(selectedItems, item);
    onSelectionChange?.(selectedItems);
  }

  function handleSearchInput(event: Event) {
    const target = event.target as HTMLInputElement;
    searchQuery = target.value;
    onSearchChange?.(searchQuery);
  }

  function focusSearchInput() {
    if (detailsElement?.open) {
      searchInputElement?.focus();
    }
  }

  function closeDropdown() {
    if (detailsElement) {
      detailsElement.open = false;
    }
  }

  // Handle click outside to close dropdown
  $effect(() => {
    const el = detailsElement;
    if (!el) return;

    const handleClickOutside = (event: MouseEvent) => {
      if (el && !el.contains(event.target as Node)) {
        el.open = false;
      }
    };

    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  });

  onMount(() => {
    if (!enableKeyboardShortcuts) return;

    const keyboardHandler = createSearchKeyboardHandler({
      onFocusKey: focusSearchInput,
      onEscapeKey: closeDropdown,
      shouldIgnoreWhenInputFocused: false,
      enableEscapeToBlur,
    });

    window.addEventListener("keydown", keyboardHandler);

    return () => {
      window.removeEventListener("keydown", keyboardHandler);
    };
  });
</script>

<details class="relative" bind:this={detailsElement}>
  <summary
    class="flex items-center justify-between cursor-pointer p-3 bg-ctp-surface0/20 border border-ctp-surface0/30 hover:bg-ctp-surface0/30 transition-colors text-sm"
  >
    <span class="text-ctp-text">
      {summaryDisplayText()}
    </span>
    <ChevronDown size={16} class="text-ctp-subtext0" />
  </summary>

  <div
    class="absolute top-full left-0 right-0 mt-1 bg-ctp-mantle border border-ctp-surface0/30 shadow-lg z-10 max-h-60 overflow-y-auto"
  >
    <!-- Search filter -->
    <div class="p-2 border-b border-ctp-surface0/20">
      <div
        class="flex items-center bg-ctp-surface0/20 border border-ctp-surface0/30 focus-within:ring-1 focus-within:ring-ctp-lavender focus-within:border-ctp-lavender transition-all"
      >
        <span class="text-ctp-subtext0 font-mono text-xs px-2 py-1">/</span>
        <input
          bind:this={searchInputElement}
          type="search"
          {placeholder}
          value={searchQuery}
          oninput={handleSearchInput}
          class="flex-1 bg-transparent border-0 py-1 pr-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-base"
        />
      </div>
    </div>

    <!-- Control buttons -->
    {#if showControlButtons}
      <div class="flex gap-2 p-2 border-b border-ctp-surface0/20">
        <button
          onclick={selectAllItems}
          class="px-2 py-1 text-xs bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-green hover:bg-ctp-green/10 hover:border-ctp-green/30 transition-all"
        >
          all
        </button>
        <button
          onclick={clearAllItems}
          class="px-2 py-1 text-xs bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30 transition-all"
        >
          clear
        </button>
      </div>
    {/if}

    <!-- Item checkboxes -->
    <div class="p-1">
      {#each filteredItems as item (getItemKey(item))}
        <label
          class="flex items-center gap-2 p-1 hover:bg-ctp-surface0/20 cursor-pointer text-sm"
        >
          <input
            type="checkbox"
            checked={isItemSelected(selectedItems, item)}
            onchange={() => toggleItem(item)}
            class="text-ctp-lavender focus:ring-ctp-lavender focus:ring-1 w-3 h-3"
          />
          <span class="text-ctp-text">{getItemText(item)}</span>
        </label>
      {/each}

      {#if filteredItems.length === 0}
        <div class="p-2 text-xs text-ctp-subtext0 text-center">
          no items found
        </div>
      {/if}
    </div>
  </div>
</details>
