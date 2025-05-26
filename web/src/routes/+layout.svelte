<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import WorkspaceSwitcher from "$lib/components/workspace-switcher.svelte";
  import { goto } from "$app/navigation";

  let { data, children } = $props();
  let { supabase, session, user, currentWorkspace, userWorkspaces } =
    $derived(data);

  onMount(() => {
    const { data } = supabase.auth.onAuthStateChange((_, newSession) => {
      if (newSession?.expires_at !== session?.expires_at) {
        invalidate("supabase:auth");
      }
    });

    return () => data.subscription.unsubscribe();
  });
</script>

{#if user}
  <header class="sticky top-0 z-30">
    <nav
      class="px-4 sm:px-6 py-3 sm:py-4 flex flex-row justify-between items-center bg-ctp-mantle border-b border-ctp-surface0"
    >
      <!-- Logo -->
      <button
        class="w-32 text-ctp-mauve fill-current"
        onclick={() => goto("/")}
      >
        <Logo />
      </button>
      <div class="flex items-center gap-2 sm:gap-3">
        {#if session && currentWorkspace}
          <WorkspaceSwitcher
            bind:currentWorkspace
            workspaces={userWorkspaces}
          />
        {/if}
      </div>
    </nav>
  </header>
{/if}

<main class="flex-1 w-full p-4">{@render children()}</main>
