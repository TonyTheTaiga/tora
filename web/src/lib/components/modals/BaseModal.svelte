<script lang="ts">
  import { onMount, onDestroy } from "svelte";

  let {
    title,
    children,
  }: {
    title: string;
    children: any;
  } = $props();

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });
</script>

<div
  class="fixed inset-0 bg-ctp-mantle/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden font-mono"
>
  <div
    class="w-full max-w-xl bg-ctp-mantle border border-ctp-surface0/30 overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/20"
    >
      <div class="flex items-center gap-3">
        <div class="w-1 h-5 bg-ctp-blue"></div>
        <h3 id="modal-title" class="text-lg font-bold text-ctp-text">
          {title}
        </h3>
      </div>
    </div>

    <div class="px-4 md:px-6 py-4">
      {@render children()}
    </div>
  </div>
</div>
