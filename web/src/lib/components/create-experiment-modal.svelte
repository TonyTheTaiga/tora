<script lang="ts">
  import CreateExperimentForm from "./create-experiment-form.svelte";
  import { X } from "lucide-svelte";

  let {
    toggleIsOpen,
  }: {
    toggleIsOpen: () => void;
  } = $props();
</script>

<div
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-4 z-50"
>
  <div
    class="bg-ctp-mantle w-full max-w-xl rounded-xl border border-ctp-surface0 shadow-2xl overflow-hidden"
  >
    <CreateExperimentForm {toggleIsOpen} />
  </div>
</div>
