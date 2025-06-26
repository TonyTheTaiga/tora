<script lang="ts">
  import StarfieldSimple from "$lib/components/starfield-simple.svelte";
  import Logo from "$lib/components/logo_assets/logo.svelte";
  import { goto } from "$app/navigation";
</script>

<StarfieldSimple />

<div
  class="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)] font-mono p-4"
>
  <div class="w-full max-w-3xl mx-auto">
    <div
      class="bg-ctp-surface0/10 border border-ctp-surface0/20 overflow-hidden rounded-lg"
    >
      <div
        class="p-6 sm:p-8 min-h-[240px] flex flex-col items-center justify-center text-center"
      >
        <button
          class="fill-ctp-blue w-full max-w-[12rem] mb-6"
          onclick={() => goto("/")}
        >
          <Logo />
        </button>
        <h1 class="text-xl md:text-2xl text-ctp-text font-bold mb-2">
          Documentation Coming Soon
        </h1>
        <p class="text-ctp-subtext0 text-sm max-w-md">brb w some docs o7</p>
      </div>
    </div>
  </div>
</div>
