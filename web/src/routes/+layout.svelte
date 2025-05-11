<script>
  import "../app.css";
  import { invalidate } from "$app/navigation";
  import { onMount } from "svelte";
  import Logo from "$lib/components/logo.svelte";
  import { goto } from "$app/navigation";

  let { data, children } = $props();
  let { session, supabase } = $derived(data);

  onMount(() => {
    const { data } = supabase.auth.onAuthStateChange((_, newSession) => {
      if (newSession?.expires_at !== session?.expires_at) {
        console.log("invalidating supabase:auth");
        invalidate("supabase:auth");
      }
    });

    return () => data.subscription.unsubscribe();
  });
</script>

<header>
  <nav
    class="px-6 py-4 flex flex-row justify-between bg-ctp-mantle border-b border-ctp-surface0"
  >
    <div class="w-32 lg:w-42 text-ctp-mauve fill-current">
      <Logo />
    </div>
    <button
      class="border border-ctp-blue rounded-md text-ctp-text w-28"
      onclick={() => goto("/auth")}
    >
      Sign Up
    </button>
  </nav>
</header>

<main class="m-4 h-full">
  {@render children()}
</main>
