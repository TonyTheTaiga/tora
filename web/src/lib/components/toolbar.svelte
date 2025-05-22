<script lang="ts">
  import type { Experiment } from "$lib/types";
  import {
    Minimize2,
    Eye,
    Sparkle,
    RefreshCw,
    Plus,
    User,
  } from "lucide-svelte";
  import ThemeToggle from "./theme-toggle.svelte";
  import { goto } from "$app/navigation";
  import { page } from "$app/state";

  let { selectedExperiment = $bindable(), isOpenCreate = $bindable() } =
    $props();

  let { session } = $derived(page.data);
</script>

<div
  id="toolbar"
  class="
    fixed
    bottom-12
    left-1/2
    transform -translate-x-1/2
    flex flex-row items-center gap-1
    bg-ctp-surface1 border border-ctp-surface2
    rounded-lg shadow-md z-40
    sm:scale-100 sm:opacity-100
    md:scale-120 md:hover:scale-140 md:transition-transform md:duration-300
    md:opacity-80 md:hover:opacity-100
    lg:scale-140 lg:hover:scale-160 lg:transition-transform lg:duration-300
    lg:opacity-80 lg:hover:opacity-100
  "
>
  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-colors"
    title="Create a new experiment"
    onclick={() => {
      isOpenCreate = true;
    }}
  >
    <Plus size={16} />
  </button>

  <ThemeToggle />

  {#if session && session.user}
    <button
      class="p-1.5 text-ctp-subtext0 hover:text-ctp-text"
      title="Go to user profile"
      onclick={() => {
        goto(`/users/${session.user.id}`);
      }}
    >
      <User size={16} />
    </button>
  {/if}
</div>
