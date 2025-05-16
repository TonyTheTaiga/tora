<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import { CircleUserRound, LogOut, Bean } from "lucide-svelte";
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
    class="px-4 py-4 flex flex-row justify-between items-center bg-ctp-mantle border-b border-ctp-surface0"
  >
    <button
      class="w-32 lg:w-42 text-ctp-mauve fill-current"
      onclick={() => goto("/")}
    >
      <Logo />
    </button>
    {#if session}
      <div class="flex items-center gap-4">
        <button
          class="flex items-center gap-2 px-3 py-1.5 border border-ctp-blue rounded-md text-ctp-blue hover:bg-ctp-lavender hover:text-ctp-crust transition-colors"
          onclick={() => goto(`/users/${session.user.id}`)}
        >
          <CircleUserRound size={18} />
          <span>Profile</span>
        </button>
        <form action="/auth?/logout" method="POST">
          <button
            type="submit"
            class="flex items-center gap-2 px-3 py-1.5 border border-ctp-red rounded-md text-ctp-red hover:bg-ctp-red hover:text-ctp-crust transition-colors"
            aria-label="Log out"
          >
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </form>
      </div>
    {:else}
      <button
        class="flex items-center gap-2 px-3 py-1.5 border border-ctp-blue rounded-md text-ctp-blue hover:bg-ctp-lavender hover:text-ctp-crust transition-colors"
        onclick={() => goto("/auth")}
      >
        <Bean size={18} />
        <span>Sign Up</span>
      </button>
    {/if}
  </nav>
</header>

<main class="p-4 h-full w-full">
  {@render children()}
</main>
