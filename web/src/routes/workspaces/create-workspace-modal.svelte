<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { closeCreateWorkspaceModal } from "$lib/state/app.svelte.js";
  import { enhance } from "$app/forms";
  import { goto } from "$app/navigation";

  onMount(() => {
    document.body.classList.add("overflow-hidden");
  });

  onDestroy(() => {
    document.body.classList.remove("overflow-hidden");
  });
</script>

<div
  class="fixed inset-0 bg-ctp-mantle/90 backdrop-blur-sm
         flex items-center justify-center p-4 z-50 overflow-hidden font-mono"
>
  <div
    class="w-full max-w-md bg-ctp-mantle border border-ctp-surface0/30 overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between p-4 border-b border-ctp-surface0/20"
    >
      <div class="flex items-center gap-3">
        <div class="w-2 h-6 bg-ctp-blue rounded-full"></div>
        <div>
          <h3 id="modal-title" class="text-lg font-bold text-ctp-text">
            New Workspace
          </h3>
          <div class="text-sm text-ctp-subtext0">create workspace config</div>
        </div>
      </div>
    </div>

    <form
      method="POST"
      action="?/createWorkspace"
      class="p-4 space-y-4"
      use:enhance={() => {
        return async ({ result, update }) => {
          console.log(result);
          if (result.type === "redirect") {
            goto(result.location);
          } else if (result.type === "success") {
            await update();
            closeCreateWorkspaceModal();
          }
        };
      }}
    >
      <div class="space-y-3">
        <div class="space-y-1 text-sm overflow-hidden">
          <div class="grid grid-cols-[auto_auto_1fr] gap-1 items-center">
            <span class="text-ctp-subtext0">name</span>
            <span class="text-ctp-text">=</span>
            <input
              name="name"
              type="text"
              class="bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm min-w-0"
              placeholder="workspace name"
              required
            />
          </div>
          <div class="grid grid-cols-[auto_auto_1fr] gap-1 items-start">
            <span class="text-ctp-subtext0">desc</span>
            <span class="text-ctp-text">=</span>
            <textarea
              name="description"
              rows="2"
              class="bg-ctp-surface0/20 border border-ctp-surface0/30 px-2 py-1 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all resize-none text-sm min-w-0"
              placeholder="description"
            ></textarea>
          </div>
        </div>
      </div>

      <div
        class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20"
      >
        <button
          onclick={() => {
            closeCreateWorkspaceModal();
          }}
          type="button"
          class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text px-3 py-2 text-sm transition-all"
        >
          cancel
        </button>
        <button
          type="submit"
          class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-3 py-2 text-sm transition-all"
        >
          create
        </button>
      </div>
    </form>
  </div>
</div>
