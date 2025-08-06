<script lang="ts">
  import { onMount } from "svelte";
  import { createSearchKeyboardHandler } from "./search-utils.js";

  interface Props {
    value?: string;
    placeholder?: string;
    id?: string;
    class?: string;
    maxWidth?: string;
    enableKeyboardShortcuts?: boolean;
    showSlashPrefix?: boolean;
    focusKey?: string;
    focusKeyRequiresCtrl?: boolean;
    enableEscapeToBlur?: boolean;
    onInput?: (value: string) => void;
  }

  let {
    value = $bindable(""),
    placeholder = "search...",
    id,
    class: className = "",
    maxWidth = "max-w-lg",
    enableKeyboardShortcuts = true,
    showSlashPrefix = true,
    focusKey = "f",
    focusKeyRequiresCtrl = true,
    enableEscapeToBlur = true,
    onInput,
  }: Props = $props();

  let inputElement: HTMLInputElement | null = $state(null);

  const handleInput = (event: Event) => {
    const target = event.target as HTMLInputElement;
    value = target.value;
    onInput?.(target.value);
  };

  const focusInput = () => {
    inputElement?.focus();
  };

  onMount(() => {
    if (!enableKeyboardShortcuts) return;

    const keyboardHandler = createSearchKeyboardHandler({
      onFocusKey: focusInput,
      searchInputSelector: id ? `#${id}` : 'input[type="search"]',
      focusKey,
      focusKeyRequiresCtrl,
      enableEscapeToBlur,
    });

    window.addEventListener("keydown", keyboardHandler);

    return () => {
      window.removeEventListener("keydown", keyboardHandler);
    };
  });
</script>

<div class="px-6 py-4">
  <div class={maxWidth}>
    <div
      class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all {className}"
    >
      {#if showSlashPrefix}
        <span class="text-ctp-subtext0 font-mono text-sm px-4 py-3">/</span>
      {/if}
      <input
        bind:this={inputElement}
        {id}
        type="search"
        {placeholder}
        {value}
        oninput={handleInput}
        class="flex-1 bg-transparent border-0 py-3 {showSlashPrefix
          ? 'pr-4'
          : 'px-4'} text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-base"
      />
    </div>
  </div>
</div>
