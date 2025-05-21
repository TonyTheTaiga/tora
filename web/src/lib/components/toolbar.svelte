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
      if (selectedExperiment) {
        const currentId = selectedExperiment.id;

        // Get current viewport scroll position and element position
        const scrollY = window.scrollY;
        const currentElement = document.getElementById(
          `experiment-${currentId}`,
        );
        const currentRect = currentElement?.getBoundingClientRect();

        // Calculate the element's absolute position
        const absoluteTop = currentRect ? currentRect.top + scrollY : null;

        // First minimize the experiment
        selectedExperiment = null;

        // Then after the DOM updates, scroll to the minimized card
        requestAnimationFrame(() => {
          const element = document.getElementById(`experiment-${currentId}`);
          if (element && absoluteTop !== null) {
            // Get the new position of the minimized element
            const newRect = element.getBoundingClientRect();

            // Calculate new scroll position to keep relative viewport position
            const newScrollY =
              newRect.top +
              window.scrollY -
              window.innerHeight / 2 +
              newRect.height / 2;

            // Scroll to the calculated position
            window.scrollTo({
              top: newScrollY,
              behavior: "smooth",
            });

            // Add highlight class to the minimized card
            const minimizedElement = document.getElementById(
              `minimized-${currentId}`,
            );
            if (minimizedElement) {
              minimizedElement.classList.add("shadow-highlight", "focus-ring");
              setTimeout(() => {
                minimizedElement.classList.remove(
                  "shadow-highlight",
                  "focus-ring",
                );
              }, 2000);
            }
          }
        });
      }
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
