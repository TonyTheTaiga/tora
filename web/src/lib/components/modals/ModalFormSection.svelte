<script lang="ts">
  let {
    title,
    children,
  }: {
    title: string;
    children: any;
  } = $props();
</script>

<div class="subtle-surface p-3">
  <div class="text-base text-ctp-text font-medium mb-3">
    {title}
  </div>
  <div class="space-y-3">
    {@render children()}
  </div>
</div>
