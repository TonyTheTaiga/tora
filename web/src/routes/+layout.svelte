<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import { CircleUserRound } from "lucide-svelte";
  import { goto } from "$app/navigation";

  let { data, children } = $props();
  let { session, supabase } = $derived(data);

  onMount(() => {
    const { data } = supabase.auth.onAuthStateChange((_, newSession) => {
      if (newSession?.expires_at !== session?.expires_at) {
        invalidate("supabase:auth");
      }
    });

    return () => data.subscription.unsubscribe();
  });
</script>

<header>
  <nav
    class="px-6 py-4 flex flex-row justify-between items-center bg-ctp-mantle border-b border-ctp-surface0"
  >
    <button
      class="w-32 lg:w-42 text-ctp-mauve fill-current"
      onclick={() => goto("/")}
    >
      <Logo />
    </button>
    {#if session}
      <button
        class="text-ctp-flamingo"
        onclick={() => goto(`/users/${session.user.id}`)}
      >
        <CircleUserRound size={32} />
      </button>
    {:else}
      <button
        class="border border-ctp-blue rounded-md text-ctp-text w-28 h-10"
        onclick={() => goto("/auth")}
      >
        Sign Up
      </button>
    {/if}
  </nav>
</header>

<main class="w-full p-4">
  {@render children()}
</main>
