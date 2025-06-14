<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";
  import { Briefcase, TestTube } from "lucide-svelte";
  
  interface Props {
    experiments: Experiment[];
    workspaces: Workspace[];
  }
  
  let { experiments, workspaces }: Props = $props();
  
  let activeTab = $state<'experiments' | 'workspaces'>('experiments');
  
  function formatDate(date: Date): string {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) {
      return "Today";
    } else if (diffDays === 1) {
      return "Yesterday";
    } else if (diffDays < 7) {
      return `${diffDays} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  }
</script>

<div class="bg-ctp-surface0/30 backdrop-blur-sm border border-ctp-surface1/20 rounded-lg">
  <div class="p-4 border-b border-ctp-surface1/20">
    <div class="flex items-center gap-4">
      <button
        onclick={() => activeTab = 'experiments'}
        class="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200 {activeTab === 'experiments' ? 'bg-ctp-blue/20 text-ctp-blue' : 'text-ctp-subtext1 hover:text-ctp-text hover:bg-ctp-surface0/50'}"
      >
        <TestTube size={16} />
        Recent Experiments
      </button>
      <button
        onclick={() => activeTab = 'workspaces'}
        class="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200 {activeTab === 'workspaces' ? 'bg-ctp-blue/20 text-ctp-blue' : 'text-ctp-subtext1 hover:text-ctp-text hover:bg-ctp-surface0/50'}"
      >
        <Briefcase size={16} />
        Recent Workspaces
      </button>
    </div>
  </div>
  
  <div class="max-h-80 overflow-y-auto">
    {#if activeTab === 'experiments'}
      {#each experiments as experiment}
        <div class="p-4 border-b border-ctp-surface0/50 last:border-b-0 hover:bg-ctp-surface0/30 transition-colors duration-200">
          <div class="flex items-start justify-between">
            <div class="min-w-0 flex-1">
              <h4 class="text-sm font-medium text-ctp-text truncate">
                {experiment.name}
              </h4>
              {#if experiment.description}
                <p class="text-xs text-ctp-subtext0 mt-1 line-clamp-2">
                  {experiment.description}
                </p>
              {/if}
              <div class="flex items-center gap-3 mt-2">
                <span class="text-xs text-ctp-subtext1">
                  {formatDate(experiment.createdAt)}
                </span>
                {#if experiment.tags.length > 0}
                  <div class="flex gap-1">
                    {#each experiment.tags.slice(0, 2) as tag}
                      <span class="px-2 py-0.5 text-xs bg-ctp-blue/20 text-ctp-blue rounded">
                        {tag}
                      </span>
                    {/each}
                    {#if experiment.tags.length > 2}
                      <span class="text-xs text-ctp-subtext0">
                        +{experiment.tags.length - 2}
                      </span>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
            <div class="flex-shrink-0 ml-4">
              <span class="px-2 py-1 text-xs bg-ctp-green/20 text-ctp-green rounded">
                {experiment.visibility}
              </span>
            </div>
          </div>
        </div>
      {/each}
      
      {#if experiments.length === 0}
        <div class="p-6 text-center text-ctp-subtext0">
          <TestTube size={32} class="mx-auto mb-2 opacity-50" />
          <p class="text-sm">No recent experiments</p>
          <p class="text-xs mt-1">Create your first experiment to get started</p>
        </div>
      {/if}
    {:else}
      {#each workspaces as workspace}
        <a href="/workspaces/{workspace.id}" class="block p-4 border-b border-ctp-surface0/50 last:border-b-0 hover:bg-ctp-surface0/30 transition-colors duration-200">
          <div class="flex items-center justify-between">
            <div class="min-w-0 flex-1">
              <div class="text-sm font-medium text-ctp-text">{workspace.name}</div>
              {#if workspace.description}
                <div class="text-xs text-ctp-subtext0 mt-1 truncate">{workspace.description}</div>
              {/if}
              <div class="text-xs text-ctp-subtext1 mt-2">
                {formatDate(workspace.createdAt)}
              </div>
            </div>
            <div class="flex-shrink-0 ml-4">
              <span class="px-2 py-1 text-xs bg-ctp-mauve/20 text-ctp-mauve rounded capitalize">
                {workspace.role.toLowerCase()}
              </span>
            </div>
          </div>
        </a>
      {/each}
      
      {#if workspaces.length === 0}
        <div class="p-6 text-center text-ctp-subtext0">
          <Briefcase size={32} class="mx-auto mb-2 opacity-50" />
          <p class="text-sm">No workspaces available</p>
          <p class="text-xs mt-1">Create or join a workspace to get started</p>
        </div>
      {/if}
    {/if}
  </div>
</div>