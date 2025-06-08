<script lang="ts">
  import { onMount } from "svelte";
  import type { PageData } from "./$types";
  import { Circle } from "lucide-svelte";
  import { reset } from "$lib/components/comparison/state.svelte";
  reset();

  let { data }: { data: PageData } = $props();

  const catppuccinAccentNames = [
    "--color-ctp-red",
    "--color-ctp-peach",
    "--color-ctp-yellow",
    "--color-ctp-green",
    "--color-ctp-teal",
    "--color-ctp-sky",
    "--color-ctp-sapphire",
    "--color-ctp-blue",
    "--color-ctp-lavender",
    "--color-ctp-mauve",
    "--color-ctp-pink",
    "--color-ctp-flamingo",
    "--color-ctp-rosewater",
    "--color-ctp-maroon",
  ];

  function generateColorVariant(baseColor: string, variant: number): string {
    const match = baseColor.match(/(\d+)/g);
    if (!match || match.length < 3) return baseColor;

    const r = parseInt(match[0]) / 255;
    const g = parseInt(match[1]) / 255;
    const b = parseInt(match[2]) / 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0,
      s = 0,
      l = (max + min) / 2;

    if (max !== min) {
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

    const saturationAdjust = variant === 1 ? 0.8 : variant === 2 ? 1.2 : 0.9;
    const lightnessAdjust = variant === 1 ? 1.1 : variant === 2 ? 0.9 : 1.05;

    s = Math.min(1, s * saturationAdjust);
    l = Math.min(1, l * lightnessAdjust);

    return `hsl(${Math.round(h * 360)}, ${Math.round(s * 100)}%, ${Math.round(l * 100)}%)`;
  }

  let experimentColors_map = $state(new Map<string, string>());

  let hyperparams = $derived.by(() => {
    const keys = new Set<string>();
    data.experiments?.forEach((exp) =>
      exp.hyperparams?.forEach((hp) => keys.add(hp.key)),
    );
    return Array.from(keys).sort((a, b) => a.localeCompare(b));
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

  onMount(() => {
    const resolveAndSetColors = () => {
      if (!data.experiments) return;

      const computedStyles = getComputedStyle(document.documentElement);
      const newColorMap = new Map<string, string>();
      const resolvedBaseColors = catppuccinAccentNames.map((name) =>
        computedStyles.getPropertyValue(name).trim(),
      );

      data.experiments.forEach((exp, index) => {
        const numBaseColors = resolvedBaseColors.length;

        if (index < numBaseColors) {
          newColorMap.set(exp.id, resolvedBaseColors[index]);
        } else {
          const baseIndex = (index - numBaseColors) % numBaseColors;
          const variant =
            Math.floor((index - numBaseColors) / numBaseColors) + 1;
          const resolvedBaseColor = resolvedBaseColors[baseIndex];
          const variantColor = generateColorVariant(resolvedBaseColor, variant);
          newColorMap.set(exp.id, variantColor);
        }
      });

      experimentColors_map = newColorMap;
    };

    resolveAndSetColors();
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener("change", resolveAndSetColors);
    return () => {
      mediaQuery.removeEventListener("change", resolveAndSetColors);
    };
  });
</script>

<div class="text-ctp-text">
  <div class="mb-4">
    <!-- Legend -->
    <div class="mb-4 p-3 bg-ctp-mantle/50 rounded-md">
      <h4 class="text-xs font-medium text-ctp-subtext1 mb-3">Legend</h4>
      <div class="flex flex-wrap gap-x-4 gap-y-2">
        {#each data.experiments as experiment}
          <div class="flex items-center gap-2">
            <Circle
              size={10}
              style="color: {experimentColors_map.get(
                experiment.id,
              )}; fill: {experimentColors_map.get(experiment.id)};"
              class="flex-shrink-0"
            />
            <span class="text-xs text-ctp-text" title={experiment.name}
              >{experiment.name}</span
            >
          </div>
        {/each}
      </div>
    </div>
  </div>

  <div class="border border-ctp-surface0 rounded-md bg-ctp-base">
    <!-- hyperparams -->
    <div
      class="text-xs font-medium text-ctp-subtext1 p-2 bg-ctp-mantle border-b border-ctp-surface0 rounded"
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
