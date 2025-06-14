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
  <a
    {href}
    class="block p-4 sm:p-6 bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 rounded-2xl shadow-xl hover:bg-ctp-surface0/20 hover:scale-105 transition-all duration-200 active:scale-95"
  >
    <div class="text-xs font-medium text-ctp-subtext0 mb-1">{title}</div>
    <div class="text-2xl font-bold text-ctp-text mb-1">{value}</div>
    {#if description}
      <div class="text-xs text-ctp-subtext1">{description}</div>
    {/if}
  </a>
{:else}
  <div
    class="p-4 sm:p-6 bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 rounded-2xl shadow-xl"
  >
    <div class="text-xs font-medium text-ctp-subtext0 mb-1">{title}</div>
    <div class="text-2xl font-bold text-ctp-text mb-1">{value}</div>
    {#if description}
      <div class="text-xs text-ctp-subtext1">{description}</div>
    {/if}
  </div>
{/if}
