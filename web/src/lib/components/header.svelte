<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
  import { getTheme, toggleTheme } from "$lib/state/theme.svelte";
  import { Cog, Moon, Sun, House, Globe, Menu } from "@lucide/svelte";
  import { DropdownMenu } from "bits-ui";
  let theme = $derived(getTheme());
</script>

<header class="shrink-0 sticky top-0 z-30 surface-glass-elevated">
  <nav class="px-6 py-4 flex flex-row justify-between items-center">
    <a href="/dashboard" class="w-32 text-ctp-blue fill-current block">
      <Logo />
    </a>

    <div class="flex space-x-4">
      <DropdownMenu.Root>
        <DropdownMenu.Trigger
          class="menu-trigger floating-element p-2 rounded-none"
          aria-label="Open menu"
        >
          <Menu size={20} />
        </DropdownMenu.Trigger>
        <DropdownMenu.Portal>
          <DropdownMenu.Content align="end" class="menu-content text-sm">
            <DropdownMenu.Group>
              <DropdownMenu.Item
                class="menu-item flex items-center gap-2"
                onSelect={() => goto("/")}
              >
                <House size={16} />
                <span>Home</span>
              </DropdownMenu.Item>

              <DropdownMenu.Item
                class="menu-item flex items-center gap-2 justify-between"
                disabled
              >
                <div class="flex items-center gap-2">
                  <Globe size={16} />
                  <span>Globe</span>
                </div>
                <span
                  class="text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded-sm bg-ctp-surface0 text-ctp-subtext0"
                >
                  Soon
                </span>
              </DropdownMenu.Item>

              <DropdownMenu.Item
                class="menu-item flex items-center gap-2"
                onSelect={() => toggleTheme()}
              >
                {#if theme === "dark"}
                  <Sun size={16} />
                  <span>Light Mode</span>
                {:else}
                  <Moon size={16} />
                  <span>Dark Mode</span>
                {/if}
              </DropdownMenu.Item>

              <DropdownMenu.Item
                class="menu-item flex items-center gap-2"
                onSelect={() => goto("/settings")}
              >
                <Cog size={16} />
                <span>Settings</span>
              </DropdownMenu.Item>
            </DropdownMenu.Group>
          </DropdownMenu.Content>
        </DropdownMenu.Portal>
      </DropdownMenu.Root>
    </div>
  </nav>
</header>

<style>
  :global(.menu-content) {
    background: var(--color-ctp-base);
    border: 1px solid var(--color-ctp-surface0);
    backdrop-filter: blur(12px);
    color: var(--color-ctp-text);
    min-width: 12rem;
    outline: none;
    padding: 0.25rem;
    border-radius: 0;
    box-shadow:
      0 10px 15px -3px rgb(0 0 0 / 0.1),
      0 4px 6px -4px rgb(0 0 0 / 0.1);
    z-index: 50;
  }

  :global(.menu-content:focus),
  :global(.menu-content:focus-visible) {
    outline: none;
    box-shadow:
      0 10px 15px -3px rgb(0 0 0 / 0.1),
      0 4px 6px -4px rgb(0 0 0 / 0.1);
  }

  :global(.menu-item) {
    color: var(--color-ctp-text);
    padding: 0.5rem 0.625rem;
    transition:
      background-color 0.25s ease,
      color 0.25s ease;
    border: none;
    outline: none;
    border-radius: 0;
    cursor: pointer;
  }

  :global(.menu-item:hover) {
    background: var(--color-ctp-surface0);
    color: var(--color-ctp-text);
  }

  :global(.menu-item:focus),
  :global(.menu-item:focus-visible) {
    outline: none;
    box-shadow: none;
    background: var(--color-ctp-surface0);
    color: var(--color-ctp-text);
  }

  :global(.menu-item[data-disabled]) {
    opacity: 0.5;
    pointer-events: none;
    cursor: not-allowed;
  }

  :global(.menu-trigger) {
    background: transparent;
    border: none;
    outline: none;
    cursor: pointer;
  }

  :global(.menu-trigger:focus),
  :global(.menu-trigger:focus-visible) {
    outline: none;
    box-shadow: none;
  }
</style>
