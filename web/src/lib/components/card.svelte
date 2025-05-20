<script lang="ts">
  let {
    hover = true,
    border = "none",
    borderColor = "ctp-subtext0",
    background = "bg-ctp-crust",
    opacity = "opacity-100",
    children,
  } = $props();
</script>

<article
  class="
    p-4 h-[200px] rounded-lg
    {`${background}`} 
    {`${opacity}`}
    {border !== 'none' ? `border-2 border-${border} border-${borderColor}` : ''}
  "
>
  {@render children()}
</article>
