<script lang="ts">
  import {
    GripVertical,
    Minimize2,
    Eye,
    Sparkle,
    RefreshCw,
  } from "lucide-svelte";
</script>

<div
  id="toolbar"
  class="
    fixed
    bottom-4
    left-1/2
    transform -translate-x-1/2
    flex flex-row items-center gap-1
    bg-ctp-surface1 border border-ctp-surface2
    rounded-lg shadow-md z-40
    opacity-85
  "
>
  <button
    id="grabber"
    class="flex items-center justify-center cursor-move select-none p-1.5 text-ctp-subtext0 hover:text-ctp-text"
    aria-label="Drag handle"
  >
    <GripVertical size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-colors"
    title="Minimize active experiment"
  >
    <Minimize2 size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-colors"
    title="Toggle experiment highlighting"
  >
    <Eye size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform active:rotate-90"
    title="Get AI recommendations"
  >
    <Sparkle size={16} />
  </button>

  <button
    class="p-1.5 text-ctp-subtext0 hover:text-ctp-text transition-transform hover:rotate-180 duration-300"
    title="Refresh experiments"
  >
    <RefreshCw size={16} />
  </button>
</div>
