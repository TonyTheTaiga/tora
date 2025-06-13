<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import { goto } from "$app/navigation";
  import Toolbar from "$lib/components/toolbar.svelte";

  let { data, children } = $props();
  let { supabase, session, user } = $derived(data);

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
      <button class="w-32 text-ctp-blue fill-current" onclick={() => goto("/")}>
        <Logo />
      </button>
    </nav>
  </header>
  <Toolbar />
{/if}
<main class="flex-1 w-full p-2">{@render children()}</main>
