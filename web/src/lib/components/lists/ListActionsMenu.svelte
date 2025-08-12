<script lang="ts">
  import { DropdownMenu } from "bits-ui";
  import { MoreHorizontal } from "@lucide/svelte";
  import type { Snippet } from "svelte";

  type ActionItem = {
    label: string;
    onSelect: () => void;
    icon?: any;
    destructive?: boolean;
    textValue?: string;
  };

  type SeparatorItem = { type: "separator" };

  export type MenuItem = ActionItem | SeparatorItem;

  interface Props {
    items: MenuItem[];
    ariaLabel?: string;
    align?: "start" | "center" | "end";
    trigger?: Snippet<[]>;
  }

  let {
    items,
    ariaLabel = "Open actions menu",
    align = "end",
    trigger,
  }: Props = $props();
</script>

<DropdownMenu.Root>
  <DropdownMenu.Trigger
    class="menu-trigger text-ctp-subtext1 hover:text-ctp-text p-1"
    aria-label={ariaLabel}
  >
    {#if trigger}
      {@render trigger()}
    {:else}
      <MoreHorizontal class="w-4 h-4" />
    {/if}
  </DropdownMenu.Trigger>
  <DropdownMenu.Portal>
    <DropdownMenu.Content {align} class="menu-content text-sm">
      <DropdownMenu.Group aria-label={ariaLabel}>
        {#each items as item}
          {#if (item as any).type === "separator"}
            <DropdownMenu.Separator class="menu-separator" />
          {:else}
            {#key (item as any).label}
              {@const action = item as ActionItem}
              <DropdownMenu.Item
                class={`menu-item flex items-center gap-2 ${action.destructive ? "destructive" : ""}`}
                onSelect={action.onSelect}
                textValue={action.textValue ?? action.label}
              >
                {#if action.icon}
                  <action.icon class="w-3.5 h-3.5" />
                {/if}
                <span>{action.label}</span>
              </DropdownMenu.Item>
            {/key}
          {/if}
        {/each}
      </DropdownMenu.Group>
    </DropdownMenu.Content>
  </DropdownMenu.Portal>
</DropdownMenu.Root>

<style>
  :global(.menu-content) {
    background: color-mix(
      in srgb,
      var(--color-ctp-surface0) 16%,
      var(--color-ctp-base)
    );
    border: none;
    backdrop-filter: blur(12px);
    color: var(--color-ctp-subtext0);
    min-width: 10rem;
    outline: none;
    padding: 0;
  }

  :global(.menu-content:focus),
  :global(.menu-content:focus-visible) {
    outline: none;
    box-shadow: none;
  }

  :global(.menu-item) {
    color: var(--color-ctp-subtext0);
    padding: 0.5rem 0.625rem;
    transition:
      background-color 0.25s ease,
      color 0.25s ease;
    border: none;
    outline: none;
  }

  :global(.menu-item:hover) {
    background: color-mix(
      in srgb,
      var(--color-ctp-surface0) 24%,
      var(--color-ctp-base)
    );
    color: var(--color-ctp-text);
  }

  :global(.menu-item:focus),
  :global(.menu-item:focus-visible) {
    outline: none;
    box-shadow: none;
    background: color-mix(
      in srgb,
      var(--color-ctp-surface0) 24%,
      var(--color-ctp-base)
    );
    color: var(--color-ctp-text);
  }

  :global(.menu-item.destructive) {
    color: var(--color-ctp-red);
  }

  :global(.menu-item.destructive:hover) {
    background: color-mix(
      in srgb,
      var(--color-ctp-red) 10%,
      var(--color-ctp-base)
    );
    color: var(--color-ctp-red);
  }

  :global(.menu-separator) {
    height: 1px;
    background: color-mix(in srgb, var(--color-ctp-surface0) 24%, transparent);
    margin: 0;
  }

  :global(.menu-trigger) {
    background: transparent;
    border: none;
    outline: none;
  }

  :global(.menu-trigger:focus),
  :global(.menu-trigger:focus-visible) {
    outline: none;
    box-shadow: none;
  }
</style>
