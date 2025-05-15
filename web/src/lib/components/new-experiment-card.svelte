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

<article class="p-4 rounded-lg hover:bg-ctp-surface0 transition-colors">
  {#if isUserSignedIn}
    <div class="flex flex-col items-center justify-center py-8 text-center">
      <!-- light icon ring instead of solid bg -->
      <div
        class="w-10 h-10 flex items-center justify-center rounded-full border border-ctp-mauve text-ctp-mauve mb-3"
      >
        <Plus size={20} />
      </div>
      <!-- smaller, lighter heading -->
      <h3 class="text-base font-medium text-ctp-subtext0 mb-1">
        New Experiment
      </h3>
      <p class="text-sm text-ctp-subtext0 mb-3 max-w-xs">
        Start tracking a new experiment with metrics and parameters.
      </p>
      <!-- outline button -->
      <button
        onclick={openModal}
        class="px-3 py-1 rounded-md border border-ctp-mauve text-ctp-mauve hover:bg-ctp-mauve hover:text-ctp-crust transition-colors font-medium text-sm"
      >
        Create
      </button>
    </div>
  {:else}
    <div class="flex flex-col items-center justify-center py-8 text-center">
      <div
        class="w-10 h-10 flex items-center justify-center rounded-full border border-ctp-blue text-ctp-blue mb-3"
      >
        <UserRound size={20} />
      </div>
      <h3 class="text-base font-medium text-ctp-subtext0 mb-1">
        Create an Account
      </h3>
      <p class="text-sm text-ctp-subtext0 mb-3 max-w-xs">
        Ready to level up? Sign up to track your experiments and metrics—or
        discover what others are working on.
      </p>
      <a
        href="/auth"
        class="px-3 py-1 rounded-md border border-ctp-blue text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors font-medium text-sm"
      >
        Sign Up / Login
      </a>
    </div>
  {/if}
</article>

