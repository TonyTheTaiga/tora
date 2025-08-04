import type { Workspace, Experiment } from "$lib/types";

let selectedWorkspace = $state<Workspace | null>(null);
let selectedExperiment = $state<Experiment | null>(null);

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
