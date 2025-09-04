<script lang="ts">
  import { browser } from "$app/environment";

  let { children } = $props();

  function isUserOnMobile() {
    if (!browser) return false;
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
      navigator.userAgent,
    );
  }
</script>

{#if isUserOnMobile()}
  <div class="h-full flex items-center justify-center p-6">
    <div class="text-center text-ctp-text">
      <h1 class="text-xl font-medium mb-2">Mobile App Coming Soon</h1>
      <p class="text-ctp-subtext0">Check back soon!</p>
    </div>
  </div>
{:else}
  {@render children()}
{/if}
