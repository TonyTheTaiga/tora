<script lang="ts">
  let {
    name,
    type = "text",
    placeholder,
    value = $bindable(),
    required = false,
    rows,
    ...restProps
  }: {
    name: string;
    type?: string;
    placeholder?: string;
    value?: string;
    required?: boolean;
    rows?: number;
    [key: string]: any;
  } = $props();

  const baseClasses =
    "w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm";
</script>

{#if type === "textarea"}
  <textarea
    {name}
    {placeholder}
    {required}
    {rows}
    bind:value
    class="{baseClasses} resize-none"
    {...restProps}
  ></textarea>
{:else}
  <input
    {name}
    {type}
    {placeholder}
    {required}
    bind:value
    class={baseClasses}
    {...restProps}
  />
{/if}
