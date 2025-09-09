<script lang="ts">
  let { message } = $props();
</script>

<div
  class="flex flex-col items-center justify-center text-center p-8 space-y-3"
>
  <div
    class="w-12 h-12 rounded-full bg-ctp-surface0/30 flex items-center justify-center"
  >
    <div
      class="w-6 h-6 border-2 border-ctp-subtext0/30 rounded-full border-dashed"
    ></div>
  </div>
  <p class="text-ctp-subtext0 text-sm font-medium">{message}</p>
</div>
