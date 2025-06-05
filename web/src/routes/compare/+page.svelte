<script lang="ts">
  import type { PageData } from "./$types";
  import { Circle } from "lucide-svelte";

  let { data }: { data: PageData } = $props();

  const catppuccinAccents = {
    base: [
      "#ed8796", // Red
      "#f5a97f", // Peach
      "#eed49f", // Yellow
      "#a6da95", // Green
      "#8bd5ca", // Teal
      "#91d7e3", // Sky
      "#7dc4e4", // Sapphire
      "#8aadf4", // Blue
      "#b7bdf8", // Lavender
      "#c6a0f6", // Mauve
      "#f5bde6", // Pink
      "#f0c6c6", // Flamingo
      "#f4dbd6", // Rosewater
      "#ee99a0", // Maroon
    ],
    variants: [],
  };

  function generateCatppuccinColor(index: number): string {
    const baseColors = catppuccinAccents.base;

    if (index < baseColors.length) {
      return baseColors[index];
    }

    const baseIndex = (index - baseColors.length) % baseColors.length;
    const baseColor = baseColors[baseIndex];
    const variant = Math.floor((index - baseColors.length) / baseColors.length);
    const hex = baseColor.slice(1);
    const r = parseInt(hex.slice(0, 2), 16) / 255;
    const g = parseInt(hex.slice(2, 4), 16) / 255;
    const b = parseInt(hex.slice(4, 6), 16) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    let s = 0;
    let l = (max + min) / 2;

    if (max === min) {
      h = s = 0;
    } else {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch (max) {
        case r:
          h = (g - b) / d + (g < b ? 6 : 0);
          break;
        case g:
          h = (b - r) / d + 2;
          break;
        case b:
          h = (r - g) / d + 4;
          break;
      }
      h /= 6;
    }

    const saturationAdjust = variant === 0 ? 0.8 : variant === 1 ? 1.2 : 0.9;
    const lightnessAdjust = variant === 0 ? 0.9 : variant === 1 ? 1.1 : 1.05;
    s = Math.min(1, s * saturationAdjust);
    l = Math.min(1, l * lightnessAdjust);
    return `hsl(${Math.round(h * 360)}, ${Math.round(s * 100)}%, ${Math.round(l * 100)}%)`;
  }

  let hyperparams = $derived.by(() => {
    const keys = new Set<string>();
    data.experiments?.forEach((exp) =>
      exp.hyperparams?.forEach((hp) => keys.add(hp.key)),
    );
    return keys;
  });

  let idToHP = $derived.by(() => {
    const ret = new Map();
    data.experiments?.forEach((exp) => {
      ret.set(exp.id, new Map());
      exp.hyperparams?.forEach((hp) => {
        ret.get(exp.id).set(hp.key, hp.value);
      });
    });
    return ret;
  });

  let experimentColors_map = $derived.by(() => {
    const colorMap = new Map();
    data.experiments?.forEach((exp, index) => {
      colorMap.set(exp.id, generateCatppuccinColor(index));
    });
    return colorMap;
  });
</script>

<div class="text-ctp-text">
  <div class="mb-4">
    <!-- Legend -->
    <div class="mb-4 p-3 bg-ctp-mantle border border-ctp-surface0 rounded-md">
      <h4 class="text-xs font-medium text-ctp-subtext1 mb-2">Legend</h4>
      <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
        {#each data.experiments as experiment}
          <div
            class="flex items-center gap-1.5 p-1.5 bg-ctp-surface0 rounded border border-ctp-surface1 hover:bg-ctp-surface1 transition-colors"
          >
            <Circle
              size={12}
              style="color: {experimentColors_map.get(
                experiment.id,
              )}; fill: {experimentColors_map.get(experiment.id)};"
              class="flex-shrink-0"
            />
            <span class="text-xs text-ctp-text truncate" title={experiment.name}
              >{experiment.name}</span
            >
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="border border-ctp-surface0 rounded-md bg-ctp-base">
    <div
      class="text-xs font-medium text-ctp-subtext1 p-2 bg-ctp-mantle border-b border-ctp-surface0"
    >
      Hyperparameters
    </div>
    <div
      class="overflow-x-auto overflow-y-auto max-h-44 scroll-container"
      style="scrollbar-width: none; -ms-overflow-style: none;"
    >
      <table class="w-full text-sm text-left">
        <thead class="bg-ctp-mantle sticky top-0 z-10">
          <tr>
            <th
              class="p-3 font-medium text-ctp-subtext1 border-b border-ctp-surface0 sticky left-0 bg-ctp-mantle z-20"
            >
              •
            </th>
            {#each hyperparams as hyperparam}
              <th
                class="p-3 font-medium text-ctp-subtext1 border-b border-ctp-surface0 whitespace-nowrap"
              >
                {hyperparam.toLowerCase()}
              </th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each data.experiments as experiment}
            <tr
              class="border-t border-ctp-surface0 hover:bg-ctp-surface0/30 transition-colors"
            >
              <th
                scope="row"
                class="p-3 text-ctp-text font-medium bg-ctp-mantle sticky left-0 text-center"
              >
                <Circle
                  size={16}
                  style="color: {experimentColors_map.get(
                    experiment.id,
                  )}; fill: {experimentColors_map.get(experiment.id)};"
                  class="mx-auto"
                />
              </th>
              {#each hyperparams as key}
                <td class="p-3 text-ctp-text whitespace-nowrap">
                  {idToHP.get(experiment.id).get(key) ?? "-"}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>
</div>

<style>
  .scroll-container::-webkit-scrollbar {
    display: none;
  }
</style>
