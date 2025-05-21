<script lang="ts">
  import type { Experiment } from "$lib/types";
  import { Minimize2, Eye, Sparkle, RefreshCw } from "lucide-svelte";
  import { goto } from "$app/navigation";

  let {
    selectedExperiment = $bindable(),
  }: { selectedExperiment: Experiment | null } = $props();
</script>

<div
  id="toolbar"
  class="
    fixed
    bottom-12
    left-1/2
    transform -translate-x-1/2
    flex flex-row items-center gap-1
    bg-ctp-surface1 border border-ctp-surface2
    rounded-lg shadow-md z-40
    opacity-85
  "
>
  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-colors"
    title="Minimize active experiment"
    onclick={() => {
      const detailsElements = document.querySelectorAll("details");
      detailsElements.forEach((details) => {
        details.removeAttribute("open");
      });
    }}
  >
    <Minimize2 size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-colors"
    title="Toggle experiment highlighting"
  >
    <Eye size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
    title="Get AI recommendations"
  >
    <Sparkle size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform hover:rotate-180 duration-300"
    title="Refresh experiments"
    onclick={() => {
      goto("/");
    }}
  >
    <RefreshCw size={16} />
  </button>
</div>
