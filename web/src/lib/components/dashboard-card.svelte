<script lang="ts">
  interface Props {
    title: string;
    value: string | number;
    description?: string;
    href?: string;
  }
  
  let { title, value, description, href }: Props = $props();
</script>

{#if href}
  <a {href} class="block p-4 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface1/20 rounded-lg hover:bg-ctp-surface0/50 transition-colors duration-200">
    <div class="text-xs font-medium text-ctp-subtext1 mb-1">{title}</div>
    <div class="text-2xl font-bold text-ctp-text mb-1">{value}</div>
    {#if description}
      <div class="text-xs text-ctp-subtext0">{description}</div>
    {/if}
  </a>
{:else}
  <div class="p-4 bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface1/20 rounded-lg">
    <div class="text-xs font-medium text-ctp-subtext1 mb-1">{title}</div>
    <div class="text-2xl font-bold text-ctp-text mb-1">{value}</div>
    {#if description}
      <div class="text-xs text-ctp-subtext0">{description}</div>
    {/if}
  </div>
{/if}