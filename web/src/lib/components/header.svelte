<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
</script>

<header class="sticky top-0 z-30 surface-glass-elevated">
  <nav
    class="px-4 sm:px-6 py-3 sm:py-4 flex flex-row justify-between items-center"
  >
    <button
      class="w-32 text-ctp-blue fill-current hover:text-ctp-blue/80 transition-colors"
      onclick={() => goto("/")}
    >
      <Logo />
    </button>
  </nav>
</header>
