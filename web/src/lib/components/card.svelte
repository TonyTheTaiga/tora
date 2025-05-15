<script lang="ts">
  let {
    hover = $bindable(true),
    border = $bindable("dashed"),
    borderColor = $bindable("ctp-subtext0"),
    height = $bindable("auto"),
    children,
  } = $props();
</script>

<article
  class="p-4 rounded-lg flex flex-col {hover
    ? 'hover:bg-ctp-surface0 transition-colors'
    : ''} {border !== 'none'
    ? `border border-${border} border-${borderColor}`
    : ''}"
  style="height: {height === 'auto' ? 'auto' : `${height}px`};"
>
  {@render children()}
</article>

