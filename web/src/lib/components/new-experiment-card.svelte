<script lang="ts">
  import { Plus, UserRound } from "lucide-svelte";
  import { page } from "$app/state";

  let {
    createNewExperimentFlag = $bindable(),
  }: {
    createNewExperimentFlag: boolean;
  } = $props();

  let isUserSignedIn: boolean = $state(!!page.data.session);
</script>

<article class="h-full">
  <div class="flex flex-col h-full">
    {#if isUserSignedIn}
      <!-- Header -->
      <div class="mb-2">
        <div
          class="w-8 h-8 flex items-center justify-center rounded-full border border-dashed border-ctp-mauve text-ctp-mauve"
        >
          <Plus size={16} />
        </div>
      </div>

      <!-- Content (grows to fill space) -->
      <div class="flex-grow flex flex-col">
        <h3 class="text-sm font-medium text-ctp-subtext0 mb-1">
          New Experiment
        </h3>
        <p class="text-xs text-ctp-subtext0 leading-relaxed">
          Start tracking a new experiment with metrics and parameters.
        </p>

        <!-- Spacer that grows to push button to bottom -->
        <div class="flex-grow"></div>
      </div>

      <!-- Footer -->
      <div class="mt-2">
        <button
          onclick={() => {
            createNewExperimentFlag = true;
          }}
          class="px-2.5 py-1 rounded-md border border-ctp-mauve text-ctp-mauve hover:bg-ctp-mauve hover:text-ctp-crust transition-colors font-medium text-xs"
        >
          Create
        </button>
      </div>
    {:else}
      <!-- Header -->
      <div class="mb-2">
        <div
          class="w-8 h-8 flex items-center justify-center rounded-full border border-dashed border-ctp-blue text-ctp-blue"
        >
          <UserRound size={16} />
        </div>
      </div>

      <!-- Content (grows to fill space) -->
      <div class="flex-grow flex flex-col">
        <h3 class="text-sm font-medium text-ctp-subtext0 mb-1">
          Create an Account
        </h3>
        <p class="text-xs text-ctp-subtext0 leading-relaxed">
          Sign up to start tracking your experiments and metrics or browse
          public experiments.
        </p>
      </div>

      <!-- Footer -->
      <div class="mt-2">
        <a
          href="/auth"
          class="px-2.5 py-1 rounded-md border border-ctp-blue text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors font-medium text-xs"
        >
          Sign Up / Login
        </a>
      </div>
    {/if}
  </div>
</article>
