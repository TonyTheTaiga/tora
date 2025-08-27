import type { Experiment, Workspace, ApiKey } from "$lib/types";

interface ModalState {
  createExperiment: boolean;
  editExperiment: Experiment | null;
  deleteExperiment: Experiment | null;
  selectedExperiment: Experiment | null;
  createWorkspace: boolean;
  deleteWorkspace?: Workspace | null;
  inviteWorkspace?: Workspace | null;
  revokeApiKey?: ApiKey | null;
}

interface AppState {
  modals: ModalState;
}

const state = $state<AppState>({
  modals: {
    createExperiment: false,
    editExperiment: null,
    deleteExperiment: null,
    selectedExperiment: null,
    createWorkspace: false,
    deleteWorkspace: null,
    inviteWorkspace: null,
    revokeApiKey: null,
  },
});

export function openCreateExperimentModal() {
  state.modals.createExperiment = true;
}

export function closeCreateExperimentModal() {
  state.modals.createExperiment = false;
}

export function setExperimentToEdit(experiment: Experiment) {
  state.modals.editExperiment = experiment;
}

export function resetExperimentToEdit() {
  state.modals.editExperiment = null;
}

export function setExperimentToDelete(experiment: Experiment) {
  state.modals.deleteExperiment = experiment;
}

export function resetExperimentToDelete() {
  state.modals.deleteExperiment = null;
}

export function setSelectedExperiment(experiment: Experiment | null) {
  state.modals.selectedExperiment = experiment;
}

export function getCreateExperimentModal() {
  return state.modals.createExperiment;
}

export function getExperimentToEdit() {
  return state.modals.editExperiment;
}

export function getExperimentToDelete() {
  return state.modals.deleteExperiment;
}

export function getSelectedExperiment() {
  return state.modals.selectedExperiment;
}

export function openCreateWorkspaceModal() {
  state.modals.createWorkspace = true;
}

export function closeCreateWorkspaceModal() {
  state.modals.createWorkspace = false;
}

export function getCreateWorkspaceModal() {
  return state.modals.createWorkspace;
}

// Workspace modals
export function setWorkspaceToDelete(workspace: Workspace) {
  state.modals.deleteWorkspace = workspace;
}

export function resetWorkspaceToDelete() {
  state.modals.deleteWorkspace = null;
}

export function getWorkspaceToDelete() {
  return state.modals.deleteWorkspace;
}

export function setWorkspaceToInvite(workspace: Workspace) {
  state.modals.inviteWorkspace = workspace;
}

export function resetWorkspaceToInvite() {
  state.modals.inviteWorkspace = null;
}

export function getWorkspaceToInvite() {
  return state.modals.inviteWorkspace;
}

// Settings: API key revocation
export function setApiKeyToRevoke(apiKey: ApiKey) {
  state.modals.revokeApiKey = apiKey;
}

export function resetApiKeyToRevoke() {
  state.modals.revokeApiKey = null;
}

export function getApiKeyToRevoke() {
  return state.modals.revokeApiKey;
}
