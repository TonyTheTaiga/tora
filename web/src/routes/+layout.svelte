<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Header from "$lib/components/header.svelte";
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
  <Header />
  <Toolbar />
{/if}

<main class="flex-1 w-full max-w-7xl mx-auto p-2">{@render children()}</main>
