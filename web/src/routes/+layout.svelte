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

<header class="sticky top-0 z-30">
  <nav
    class="px-4 sm:px-6 py-3 sm:py-4 flex flex-row justify-between items-center bg-ctp-mantle border-b border-ctp-surface0"
  >
    <!-- Logo -->
    <button class="w-32 text-ctp-mauve fill-current" onclick={() => goto("/")}>
      <Logo />
    </button>

    <!-- Right side actions -->
    <div class="flex items-center gap-2 sm:gap-3"></div>
  </nav>
</header>

<main class="flex-1 w-full p-4">
  {@render children()}
</main>
