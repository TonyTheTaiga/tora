<script lang="ts">
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();

  onMount(() => {
    // Only redirect on success, and give user time to read the message
    if (data.success) {
      setTimeout(() => {
        goto("/dashboard");
      }, 3000);
    }
  });
</script>

<div
  class="min-h-screen flex justify-center items-center text-center font-mono"
>
  <div class="max-w-md mx-auto p-8">
    {#if data.success}
      <div class="surface-layer-2 p-6 mb-4">
        <div class="text-ctp-green text-4xl mb-4">✓</div>
        <h1 class="text-ctp-text text-xl font-bold mb-3">Email Confirmed!</h1>
        <p class="text-ctp-subtext1 mb-4">{data.message}</p>
        <div
          class="flex items-center justify-center gap-2 text-ctp-subtext0 text-sm"
        >
          <div class="animate-pulse">Redirecting in 3 seconds...</div>
        </div>
      </div>
    {:else}
      <div class="surface-layer-2 p-6 mb-4">
        <div class="text-ctp-red text-4xl mb-4">✗</div>
        <h1 class="text-ctp-text text-xl font-bold mb-3">
          Confirmation Failed
        </h1>
        <p class="text-ctp-subtext1 mb-6">{data.message}</p>
        <div class="flex flex-col gap-3">
          <button
            onclick={() => goto("/signup")}
            class="w-full bg-ctp-blue text-ctp-crust px-4 py-2 hover:bg-ctp-blue/80 transition-colors text-sm font-medium"
          >
            Back to Sign Up
          </button>
          <button
            onclick={() => goto("/")}
            class="w-full border border-ctp-surface0 text-ctp-text px-4 py-2 hover:bg-ctp-surface0/10 transition-colors text-sm"
          >
            Go to Home
          </button>
        </div>
      </div>
    {/if}
  </div>
</div>
