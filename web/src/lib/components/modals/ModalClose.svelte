<script lang="ts">
  import { X } from "@lucide/svelte";

  let { onClose }: { onClose: () => void } = $props();
</script>

<button
  onclick={onClose}
  type="button"
  class="absolute top-3 right-3 sm:top-4 sm:right-4 md:top-6 md:right-6
         text-ctp-subtext0 hover:text-ctp-text p-1 transition-colors"
  aria-label="Close modal"
>
  <X size={20} />
</button>
