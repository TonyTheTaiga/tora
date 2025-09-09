<script lang="ts">
  import { Toolbar } from "bits-ui";
  import { RefreshCw, Baseline, Antenna } from "@lucide/svelte";

  interface Props {
    yScale: "log" | "linear";
    streaming: boolean;
    onRefresh: () => void;
    onToggleScale: () => void;
    onToggleStreaming: () => void;
  }

  let {
    yScale,
    streaming,
    onRefresh,
    onToggleScale,
    onToggleStreaming,
  }: Props = $props();
</script>

<Toolbar.Root
  class="chart-toolbar w-full border border-b-0 border-ctp-surface0/35 bg-ctp-surface0/18 shadow-sm h-10 min-w-max px-[4px] py-0 gap-2 flex items-center justify-end"
>
  <button
    class="toolbar-btn"
    onclick={() => onRefresh?.()}
    aria-label="refresh"
    type="button"
  >
    <RefreshCw class="icon" />
    <span class="label">refresh</span>
  </button>

  <button
    class="toolbar-btn"
    onclick={() => onToggleScale?.()}
    aria-label="toggle scale"
    aria-pressed={yScale === "log"}
    title="toggle Y axis scale between log and linear"
    type="button"
  >
    <Baseline class="icon" />
    <span class="label">{yScale}</span>
  </button>

  <button
    class="toolbar-btn"
    onclick={() => onToggleStreaming?.()}
    aria-label="toggle live stream"
    aria-pressed={streaming}
    type="button"
  >
    <Antenna class="icon" />
    <span class="label">live stream</span>
  </button>
</Toolbar.Root>

<style>
  :global(.chart-toolbar) {
    padding: 0; /* ensure no extra gap around toolbar */
  }
  :global(.toolbar-btn) {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem; /* 6px */
    font-size: 0.75rem; /* text-xs */
    line-height: 1rem;
    padding: 0.375rem 0.625rem; /* px-2.5 py-1.5 */
    cursor: pointer;
    background: transparent;
    border: none;
    color: var(--color-ctp-subtext0);
    transition:
      background-color 0.2s ease,
      border-color 0.2s ease,
      color 0.2s ease,
      transform 0.2s ease;
  }
  :global(.toolbar-btn:hover) {
    color: var(--color-ctp-blue);
    transform: translateY(-1px);
  }
  :global(.toolbar-btn[aria-pressed="true"]) {
    color: var(--color-ctp-blue);
  }
  :global(.toolbar-btn .icon) {
    width: 0.875rem; /* w-3.5 */
    height: 0.875rem; /* h-3.5 */
  }
  :global(.toolbar-btn .label) {
    text-transform: none;
  }
</style>
