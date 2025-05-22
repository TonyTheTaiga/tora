<script lang="ts">
  import { Plus } from "lucide-svelte";

  let {
    createNewExperimentFlag = $bindable(),
  }: {
    createNewExperimentFlag: boolean;
  } = $props();
</script>

<article class="flex flex-col h-full">
  <!-- Header -->
  <div class="mb-2">
    <div
      class="w-8 h-8 flex items-center justify-center rounded-full border border-dashed border-ctp-mauve text-ctp-mauve"
    >
      <Plus size={16} />
    </div>
  </div>

  <!-- Content -->
  <div class="flex-grow flex flex-col">
    <h3 class="text-sm font-medium text-ctp-subtext0 mb-1">New Experiment</h3>
    <p class="text-xs text-ctp-subtext0 leading-relaxed">
      Start tracking a new experiment with metrics and parameters.
    </p>
  </div>

  <!-- Footer -->
  <div class="mt-auto">
    <button
      onclick={() => {
        createNewExperimentFlag = true;
      }}
      class="px-2.5 py-1 rounded-md border border-ctp-mauve text-ctp-mauve hover:bg-ctp-mauve hover:text-ctp-crust transition-colors font-medium text-xs"
    >
      Create
    </button>
  </div>
</article>
