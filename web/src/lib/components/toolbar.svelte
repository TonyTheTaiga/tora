<script lang="ts">
  import { onMount, onDestroy, tick } from "svelte";
  import { browser } from "$app/environment";
  import {
    GripVertical,
    Minimize2,
    Eye,
    Sparkle,
    RefreshCw,
  } from "lucide-svelte";

  let pos = { x: 0, y: 0 };
  let dragging = false;
  let offset = { x: 0, y: 0 };

  let toolbarEl: HTMLDivElement;
  let parentEl: HTMLElement;
  let parentRect: DOMRect;

  function handlePointerDown(e: PointerEvent) {
    e.preventDefault();
    dragging = true;

    // get true toolbar position in viewport
    const rect = toolbarEl.getBoundingClientRect();
    offset.x = e.clientX - rect.left;
    offset.y = e.clientY - rect.top;

    parentRect = parentEl.getBoundingClientRect();

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
  }

  function handlePointerMove(e: PointerEvent) {
    if (!dragging) return;

    let newX = e.clientX - offset.x;
    let newY = e.clientY - offset.y;

    // clamp to parent bounds
    newX = Math.max(
      parentRect.left,
      Math.min(newX, parentRect.right - toolbarEl.offsetWidth),
    );
    newY = Math.max(
      parentRect.top,
      Math.min(newY, parentRect.bottom - toolbarEl.offsetHeight),
    );

    pos = { x: newX, y: newY };
  }

  function handlePointerUp() {
    dragging = false;
    window.removeEventListener("pointermove", handlePointerMove);
    window.removeEventListener("pointerup", handlePointerUp);
  }

  onMount(async () => {
    if (!browser) return;

    toolbarEl = document.getElementById("toolbar") as HTMLDivElement;
    parentEl = toolbarEl.parentElement as HTMLElement;
    await tick();
    parentRect = parentEl.getBoundingClientRect();
    pos = {
      x: parentRect.left + parentRect.width / 2 - toolbarEl.offsetWidth / 2,
      y: parentRect.top + 16,
    };
  });

  onDestroy(() => {
    window.removeEventListener("pointermove", handlePointerMove);
    window.removeEventListener("pointerup", handlePointerUp);
  });
</script>

<div
  id="toolbar"
  class="fixed flex flex-row items-center gap-1 p-0.5 bg-ctp-surface1 border border-ctp-surface2 rounded-lg shadow-md"
  style="top: {pos.y}px; left: {pos.x}px; opacity: 0.85;"
>
  <button
    id="grabber"
    class="flex items-center justify-center cursor-move select-none p-1.5 text-ctp-subtext0 hover:text-ctp-text"
    onpointerdown={handlePointerDown}
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

