<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import { closeCreateWorkspaceModal } from "$lib/state/app.svelte";
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
  class="fixed inset-0 bg-ctp-crust/80 backdrop-blur-md
         flex items-center justify-center p-2 sm:p-4 z-50 overflow-hidden"
>
  <div
    class="w-full max-w-md rounded-xl border border-ctp-surface0 shadow-2xl overflow-auto max-h-[90vh] bg-ctp-mantle"
    role="dialog"
    aria-modal="true"
    aria-labelledby="modal-title"
  >
    <div
      class="flex items-center justify-between px-6 py-4 border-b border-ctp-surface0"
    >
      <div class="flex items-center gap-2">
        <svg class="w-5 h-5 text-ctp-blue" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path>
        </svg>
        <h3 id="modal-title" class="text-xl font-medium text-ctp-text">
          Create Workspace
        </h3>
      </div>
    </div>

    <form
      method="POST"
      action="?/createWorkspace"
      class="flex flex-col gap-4 p-6"
      use:enhance={({ formElement, formData, action, cancel }) => {
        return async ({ result, update }) => {
          if (result.type === "redirect") {
            goto(result.location);
          } else if (result.type === "success") {
            closeCreateWorkspaceModal();
            window.location.reload();
          } else {
            await update();
          }
        };
      }}
    >
      <div class="flex flex-col gap-4">
        <div class="space-y-2">
          <label
            class="text-sm font-medium text-ctp-subtext0"
            for="workspace-name"
          >
            Workspace Name
          </label>
          <input
            name="workspace-name"
            type="text"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
            placeholder="Enter workspace name"
            required
          />
        </div>

        <div class="space-y-2">
          <label
            class="text-sm font-medium text-ctp-subtext0"
            for="workspace-description"
          >
            Description
          </label>
          <textarea
            name="workspace-description"
            rows="3"
            class="w-full px-3 py-2 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all resize-none placeholder-ctp-overlay0 shadow-sm"
            placeholder="Briefly describe this workspace"
          ></textarea>
        </div>
      </div>

      <div
        class="flex justify-end gap-3 pt-4 mt-2 border-t border-ctp-surface0"
      >
        <button
          onclick={() => {
            closeCreateWorkspaceModal();
          }}
          type="button"
          class="inline-flex items-center justify-center px-4 py-2 font-medium rounded-full bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          class="inline-flex items-center justify-center gap-2 px-4 py-2 font-medium rounded-full bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path>
          </svg>
          Create Workspace
        </button>
      </div>
    </form>
  </div>
</div>