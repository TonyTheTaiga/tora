<script lang="ts">
  import { Calendar } from "@lucide/svelte";
  import type { Experiment } from "$lib/types";

  let { experiment }: { experiment: Experiment } = $props();
</script>

<div class="space-y-2">
  <div class="text-sm text-ctp-text">system info</div>
  <div
    class="bg-ctp-surface0/10 border border-ctp-surface0/20 p-3 space-y-2 text-sm font-mono"
  >
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 md:gap-4">
      <div class="flex items-center gap-2">
        <Calendar size={10} class="text-ctp-subtext0" />
        <span class="text-ctp-subtext0">created_at:</span>
        <span class="text-ctp-text">
          {new Date(experiment.createdAt).toISOString()}
        </span>
      </div>
    </div>
  </div>
</div>
