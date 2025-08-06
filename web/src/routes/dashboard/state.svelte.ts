import type { Workspace, Experiment } from "$lib/types";

let selectedWorkspace = $state<Workspace | null>(null);
let selectedExperiment = $state<Experiment | null>(null);

let experimentsCache = $state(new Map<string, Experiment[]>());
let loadedWorkspaces = $state(new Set<string>());

export let loading = $state({
  workspaces: true,
  experiments: false,
  experimentDetails: false,
});

export let errors = $state({
  workspaces: null as string | null,
  experiments: null as string | null,
  experimentDetails: null as string | null,
});

export function getSelectedWorkspace(): Workspace | null {
  return selectedWorkspace;
}

export function setSelectedWorkspace(workspace: Workspace | null) {
  selectedWorkspace = workspace;
}

export function getSelectedExperiment(): Experiment | null {
  return selectedExperiment;
}

export function setSelectedExperiment(experiment: Experiment | null) {
  selectedExperiment = experiment;
}

export function getCachedExperiments(workspaceId: string): Experiment[] | null {
  return experimentsCache.get(workspaceId) || null;
}

export function setCachedExperiments(
  workspaceId: string,
  experiments: Experiment[],
): void {
  experimentsCache.set(workspaceId, experiments);
  loadedWorkspaces.add(workspaceId);
}

export function isWorkspaceLoaded(workspaceId: string): boolean {
  return loadedWorkspaces.has(workspaceId);
}

export function clearExperimentsCache(): void {
  experimentsCache.clear();
  loadedWorkspaces.clear();
}

export function clearWorkspaceCache(workspaceId: string): void {
  experimentsCache.delete(workspaceId);
  loadedWorkspaces.delete(workspaceId);
}
