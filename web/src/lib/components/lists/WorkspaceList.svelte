<script lang="ts">
  import WorkspaceRoleBadge from "$lib/components/workspace-role-badge.svelte";
  import type { Workspace } from "$lib/types";

  interface Props {
    workspaces: Workspace[];
    searchQuery?: string;
    showSummary?: boolean;
  }

  let { workspaces, searchQuery = "", showSummary = true }: Props = $props();

  let filteredWorkspaces = $derived(
    workspaces.filter((workspace) => {
      if (!searchQuery) return true;
      const query = searchQuery.toLowerCase();
      const searchableText =
        `${workspace.name} ${workspace.description || ""}`.toLowerCase();

      return query.split(" ").every((term) => searchableText.includes(term));
    }),
  );
</script>

{#if filteredWorkspaces.length === 0 && searchQuery}
  <div class="text-ctp-subtext0 text-base">
    <div>search "{searchQuery}"</div>
    <div class="text-ctp-subtext1 ml-2">no results found</div>
  </div>
{:else}
  <!-- Desktop table layout -->
  <div class="hidden md:block">
    <table class="w-full table-fixed">
      <thead>
        <tr class="text-sm text-ctp-subtext0 border-b border-ctp-surface0/20">
          <th class="text-left py-2 w-4"></th>
          <th class="text-left py-2">name</th>
          <th class="text-right py-2 w-24">role</th>
          <th class="text-right py-2 w-24">modified</th>
        </tr>
      </thead>
      <tbody>
        {#each filteredWorkspaces as workspace}
          <tr
            class="group text-base hover:bg-ctp-surface0/20 transition-colors"
          >
            <td class="py-2 px-1 w-4">
              <div class="text-ctp-green text-sm"></div>
            </td>
            <td class="py-2 px-1 min-w-0">
              <a href={`/workspaces/${workspace.id}`} class="block min-w-0">
                <div class="truncate text-ctp-text">
                  <span
                    class="group-hover:text-ctp-blue transition-colors font-medium"
                  >
                    {workspace.name}
                  </span>
                  {#if workspace.description}
                    <span class="text-ctp-subtext1 text-sm">
                      - {workspace.description}
                    </span>
                  {/if}
                </div>
              </a>
            </td>
            <td class="py-2 px-1 text-right text-sm text-ctp-subtext0 w-24">
              <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
            </td>
            <td class="py-2 px-1 text-right text-sm text-ctp-subtext0 w-24">
              {new Date(workspace.createdAt).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              })}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>

  <!-- Mobile card layout -->
  <div class="space-y-2 md:hidden">
    {#each filteredWorkspaces as workspace}
      <div
        class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 p-3 hover:bg-ctp-surface0/20 transition-colors"
      >
        <a href={`/workspaces/${workspace.id}`} class="block">
          <!-- Header row -->
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2 min-w-0 flex-1">
              <div class="text-ctp-green text-sm"></div>
              <h3
                class="text-ctp-text hover:text-ctp-blue transition-colors truncate text-base font-medium"
              >
                {workspace.name}
              </h3>
            </div>
            <div class="flex items-center gap-2 flex-shrink-0">
              <WorkspaceRoleBadge role={workspace.role || "VIEWER"} />
            </div>
          </div>

          <!-- Description -->
          {#if workspace.description}
            <p class="text-ctp-subtext1 text-sm mb-2 line-clamp-2">
              {workspace.description}
            </p>
          {/if}

          <!-- Footer with date -->
          <div
            class="flex items-center justify-between text-sm text-ctp-subtext0"
          >
            <span>
              {new Date(workspace.createdAt).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              })}
            </span>
          </div>
        </a>
      </div>
    {/each}
  </div>

  <!-- Summary line -->
  {#if showSummary}
    <div
      class="flex items-center text-sm text-ctp-subtext0 pt-2 border-t border-ctp-surface0/20 mt-4"
    >
      <div class="flex-1">
        {filteredWorkspaces.length} workspace{filteredWorkspaces.length !== 1
          ? "s"
          : ""} total
      </div>
    </div>
  {/if}
{/if}
