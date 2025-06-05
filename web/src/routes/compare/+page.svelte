<script lang="ts">
  import type { PageData } from "./$types";

  let { data }: { data: PageData } = $props();
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
</script>

<style>
  .scroll-container::-webkit-scrollbar {
    display: none;
  }
</style>

<div class="text-ctp-text">
  <div class="mb-4">
    <h3 class="text-lg font-medium text-ctp-text mb-2">
      Experiment Comparison
    </h3>
  </div>

  <div
    class="w-full overflow-x-auto overflow-y-auto max-h-48 border border-ctp-surface0 rounded-md bg-ctp-base scroll-container"
    style="scrollbar-width: none; -ms-overflow-style: none;"
  >
    <table class="w-full text-sm text-left">
      <thead class="bg-ctp-mantle sticky top-0 z-10">
        <tr>
          <th
            class="p-3 font-medium text-ctp-subtext1 border-b border-ctp-surface0 sticky left-0 bg-ctp-mantle z-20"
          >
            name
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
              class="p-3 text-ctp-text font-medium bg-ctp-mantle sticky left-0"
            >
              {experiment.name}
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
