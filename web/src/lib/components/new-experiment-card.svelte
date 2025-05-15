<script lang="ts">
  import { Plus, UserRound } from "lucide-svelte";

  let {
    isUserSignedIn = $bindable(),
    openModal = $bindable(),
  }: {
    isUserSignedIn: boolean;
    openModal: () => void;
  } = $props();
</script>

<article class="p-4 rounded-lg border border-ctp-surface1 bg-ctp-base hover:shadow-md">
  {#if isUserSignedIn}
    <!-- Logged in state -->
    <div class="flex flex-col items-center justify-center py-8 text-center">
      <div class="w-12 h-12 flex items-center justify-center rounded-full bg-ctp-mauve text-ctp-crust mb-4">
        <Plus size={24} />
      </div>
      <h3 class="text-lg font-medium text-ctp-text mb-2">New Experiment</h3>
      <p class="text-ctp-subtext0 mb-4 max-w-md">
        Start tracking a new experiment with metrics and parameters.
      </p>
      <button
        onclick={openModal}
        class="px-4 py-2 rounded-md bg-ctp-mauve text-ctp-crust hover:bg-ctp-lavender transition-colors font-medium"
      >
        Create Experiment
      </button>
    </div>
  {:else}
    <!-- Anonymous state -->
    <div class="flex flex-col items-center justify-center py-8 text-center">
      <div class="w-12 h-12 flex items-center justify-center rounded-full bg-ctp-blue text-ctp-crust mb-4">
        <UserRound size={24} />
      </div>
      <h3 class="text-lg font-medium text-ctp-text mb-2">Create an Account</h3>
      <p class="text-ctp-subtext0 mb-4 max-w-md">
        Sign up to start tracking your experiments and metrics.
      </p>
      <a
        href="/auth"
        class="px-4 py-2 rounded-md bg-ctp-blue text-ctp-crust hover:bg-ctp-lavender transition-colors font-medium"
      >
        Sign Up / Login
      </a>
    </div>
  {/if}
</article>