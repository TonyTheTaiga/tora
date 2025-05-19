<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import ThemeToggle from "$lib/components/theme-toggle.svelte";
  import { CircleUserRound, Bean } from "lucide-svelte";
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

<header class="sticky top-0 z-50">
  <nav
    class="px-4 sm:px-6 py-3 sm:py-4 flex flex-row justify-between items-center bg-ctp-mantle border-b border-ctp-surface0"
  >
    <!-- Logo -->
    <button
      class="w-28 sm:w-32 lg:w-40 text-ctp-mauve fill-current"
      onclick={() => goto("/")}
    >
      <Logo />
    </button>

    <!-- Right side actions -->
    <div class="flex items-center gap-2 sm:gap-3">
      <!-- Theme toggle -->
      <ThemeToggle />
      
      <!-- Auth actions -->
      {#if session}
        <button
          class="flex items-center gap-1.5 px-3 py-1.5 border border-ctp-blue rounded-md text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors"
          onclick={() => goto(`/users/${session.user.id}`)}
          aria-label="Profile"
          title="View profile"
        >
          <CircleUserRound size={16} class="sm:hidden" />
          <CircleUserRound size={18} class="hidden sm:inline" />
          <span class="font-medium text-xs sm:text-sm">Profile</span>
        </button>
      {:else}
        <button
          class="flex items-center gap-1.5 px-3 py-1.5 border border-ctp-blue rounded-md text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors"
          onclick={() => goto("/auth")}
          aria-label="Sign up or log in"
        >
          <Bean size={16} class="sm:hidden" />
          <Bean size={18} class="hidden sm:inline" />
          <span class="font-medium text-xs sm:text-sm">Sign Up</span>
        </button>
      {/if}
    </div>
  </nav>
</header>

<main class="p-4 h-full w-full">
  {@render children()}
</main>
