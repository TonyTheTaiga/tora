<script lang="ts">
  import { X } from "@lucide/svelte";
  import { Button } from "$lib/components";

  let { onClose }: { onClose: () => void } = $props();
</script>

<Button
  onclick={onClose}
  type="button"
  variant="icon"
  class="absolute top-6 right-6"
  aria-label="Close modal"
>
  <X size={20} />
</Button>
