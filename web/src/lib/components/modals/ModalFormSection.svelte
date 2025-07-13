<script lang="ts">
  let {
    title,
    children,
  }: {
    title: string;
    children: any;
  } = $props();
</script>

<div class="border border-ctp-surface0/20 p-3 mb-4">
  <div class="text-base text-ctp-text font-medium mb-4">
    {title}
  </div>
  <div class="space-y-3">
    {@render children()}
  </div>
</div>
