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
    class="w-full max-w-xl bg-ctp-mantle border border-ctp-surface0/30 overflow-auto max-h-[90vh]"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between p-4 md:p-6 border-b border-ctp-surface0/20"
    >
      <div class="flex items-center gap-3">
        <h3 id="modal-title" class="text-lg font-bold text-ctp-text">
          New Workspace
        </h3>
      </div>
    </div>

    <form
      method="POST"
      action="?/createWorkspace"
      class="px-4 md:px-6 py-4 space-y-4"
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
      <div class="space-y-4">
        <!-- Basic config -->
        <div class="border border-ctp-surface0/20 p-3">
          <div class="text-base text-ctp-text font-medium mb-3">
            workspace config
          </div>
          <div class="space-y-3">
            <div>
              <input
                name="name"
                type="text"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all text-sm"
                placeholder="workspace_name"
                required
              />
            </div>
            <div>
              <textarea
                name="description"
                rows="2"
                class="w-full bg-ctp-surface0/20 border border-ctp-surface0/30 px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 focus:ring-ctp-blue focus:border-ctp-blue transition-all resize-none text-sm"
                placeholder="description"
                required
              ></textarea>
            </div>
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
