<script lang="ts">
  let {
    title,
    children,
  }: {
    title: string;
    children: any;
  } = $props();
</script>

<div class="surface-layer-1 p-3">
  <div class="text-base text-ctp-mauve font-medium mb-3">
    {title}
  </div>
  <div class="space-y-3">
    {@render children()}
  </div>
</div>
