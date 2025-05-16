<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";
  import { onMount, onDestroy } from "svelte";

  let {
    isOpen = $bindable(),
  }: {
    isOpen: boolean;
  } = $props();

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });
</script>

<div
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-4 z-50"
>
  <div
    class="bg-ctp-mantle w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-hidden"
  >
    <CreateExperimentForm bind:isOpen />
  </div>
</div>
