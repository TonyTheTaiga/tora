<script lang="ts">
  let {
    hover = true,
    border = "none",
    borderColor = "ctp-subtext0",
    background = "bg-ctp-dark",
    opacity = "opacity-100",
    children,
  } = $props();
</script>

<article
  class="
    p-4 rounded-lg flex flex-col
    {`${background}`} 
    {`${opacity}`}
    {hover ? 'hover:bg-ctp-surface0 transition-colors' : ''}
    {border !== 'none' ? `border-2 border-${border} border-${borderColor}` : ''}
  "
>
  {@render children()}
</article>
