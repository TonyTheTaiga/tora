<script lang="ts">
  import type { Experiment, Workspace } from "$lib/types";
  import {
    setExperimentToEdit,
    setExperimentToDelete,
    getExperimentToDelete,
  } from "$lib/state/modal.svelte.js";
  import { Trash2, Edit } from "@lucide/svelte";
  import EmptyState from "./EmptyState.svelte";
  import ListCard from "./ListCard.svelte";
  import ListActionsMenu, { type MenuItem } from "./ListActionsMenu.svelte";
  import { DeleteExperimentModal } from "$lib/components/modals";

  interface Props {
    workspace: Workspace;
    experiments: Experiment[];
    searchQuery: string;
    onItemClick?: (experiment: Experiment) => void;
    onExperimentsChange?: (list: Experiment[]) => void;
  }

  let {
    workspace,
    experiments,
    searchQuery,
    onItemClick,
    onExperimentsChange = (list: Experiment[]) => {},
  }: Props = $props();

  function formatDate(date: Date) {
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    });
  }

  let canDeleteExperiment = $derived(["OWNER"].includes(workspace.role));

  let filteredExperiments = $derived(
    experiments
      .map((exp: Experiment) => ({
        exp,
        name: exp.name.toLowerCase(),
        desc: exp.description?.toLowerCase() ?? "",
        tags: exp.tags?.map((t: string) => t.toLowerCase()) ?? [],
      }))
      .filter(
        (entry: {
          exp: Experiment;
          name: string;
          desc: string;
          tags: string[];
        }) => {
          if (!searchQuery) return true;
          const terms = searchQuery.toLowerCase().split(/\s+/).filter(Boolean);
          return terms.every(
            (t: string) =>
              entry.name.includes(t) ||
              entry.desc.includes(t) ||
              entry.tags.some((tag: string) => tag.includes(t)),
          );
        },
      )
      .map((e: any) => e.exp),
  );

  function formatTime(date: Date): string {
    return date.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  let experimentToDelete = $derived(getExperimentToDelete());

  function openDeleteModal(experiment: Experiment) {
    setExperimentToDelete(experiment);
  }
</script>

{#if filteredExperiments.length === 0 && searchQuery}
  <EmptyState type="search" {searchQuery} />
{:else}
  <ListCard items={filteredExperiments} {onItemClick}>
    {#snippet children(experiment)}
      <h3
        class="text-ctp-text group-hover:text-ctp-blue transition-colors font-medium truncate mb-1"
      >
        {experiment.name}
      </h3>

      {#if experiment.description}
        <p class="text-ctp-subtext1 text-sm mb-2">
          {experiment.description}
        </p>
      {/if}

      {#if experiment.tags && experiment.tags.length > 0}
        <div class="flex items-center gap-2 flex-wrap mb-2">
          {#each experiment.tags.slice(0, 3) as tag}
            <span
              class="text-[11px] font-mono text-ctp-subtext1 before:content-['#'] before:text-ctp-blue/60"
              >{tag}</span
            >
          {/each}
          {#if experiment.tags.length > 3}
            <span class="text-[11px] font-mono text-ctp-overlay0">
              +{experiment.tags.length - 3}
            </span>
          {/if}
        </div>
      {/if}

      <div class="flex items-center gap-3 text-[11px] font-mono">
        {#if experiment.hyperparams && experiment.hyperparams.length > 0}
          <span class="text-ctp-subtext0">
            <span class="text-ctp-overlay0">params:</span>{experiment
              .hyperparams.length}
          </span>
        {/if}
        <span class="text-ctp-subtext0">
          <span class="text-ctp-overlay0">created:</span>{formatDate(
            experiment.createdAt,
          )}
          {formatTime(experiment.createdAt)}
        </span>
      </div>
    {/snippet}

    {#snippet actions(experiment)}
      <ListActionsMenu
        ariaLabel="experiment actions"
        items={[
          {
            label: "Edit",
            icon: Edit,
            onSelect: () => setExperimentToEdit(experiment),
          },
          ...(canDeleteExperiment
            ? [
                { type: "separator" } as const,
                {
                  label: "Delete",
                  icon: Trash2,
                  destructive: true,
                  onSelect: () => openDeleteModal(experiment),
                },
              ]
            : []),
        ] satisfies MenuItem[]}
      />
    {/snippet}
  </ListCard>
{/if}

{#if experimentToDelete}
  <DeleteExperimentModal
    experiment={experimentToDelete}
    bind:experiments
    onChange={onExperimentsChange}
  />
{/if}
