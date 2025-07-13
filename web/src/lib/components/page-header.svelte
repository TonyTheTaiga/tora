<script lang="ts">
  import type { Snippet } from "svelte";

  interface Props {
    title: string;
    subtitle?: string;
    description?: string;
    additionalInfo?: string;
    onAdditionalInfoClick?: () => void;
    additionalInfoTitle?: string;
    actionButton?: Snippet;
  }

  let {
    title,
    subtitle,
    description,
    additionalInfo,
    onAdditionalInfoClick,
    additionalInfoTitle,
    actionButton,
  }: Props = $props();
</script>

<div
  class="flex items-center justify-between p-2 sm:p-4 md:p-6 border-b border-ctp-surface0/10"
>
  <div
    class="flex items-stretch gap-2 sm:gap-3 md:gap-4 min-w-0 flex-1 pr-2 sm:pr-4 min-h-fit"
  >
    <div class="w-1 bg-ctp-blue flex-shrink-0 self-stretch"></div>
    <div class="min-w-0 flex-1 py-1">
      <h1 class="text-lg md:text-xl text-ctp-text truncate font-mono">
        {title}
      </h1>
      <div class="text-sm text-ctp-subtext0 space-y-1">
        {#if subtitle}
          <div>{subtitle}</div>
        {/if}
        {#if description}
          <div>{description}</div>
        {/if}
        {#if additionalInfo}
          {#if onAdditionalInfoClick}
            <button
              onclick={onAdditionalInfoClick}
              class="text-ctp-lavender hover:text-ctp-lavender/80 transition-colors flex text-start"
              title={additionalInfoTitle}
            >
              <span>{additionalInfo}</span>
            </button>
          {:else}
            <div>{additionalInfo}</div>
          {/if}
        {/if}
      </div>
    </div>
  </div>

  {#if actionButton}
    <div class="pl-2 sm:pl-4 md:pl-6">
      {@render actionButton()}
    </div>
  {/if}
</div>
