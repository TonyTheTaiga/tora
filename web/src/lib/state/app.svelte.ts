import type { Experiment } from "$lib/types";

interface ModalState {
  createExperiment: boolean;
  editExperiment: Experiment | null;
  deleteExperiment: Experiment | null;
  selectedExperiment: Experiment | null;
  createWorkspace: boolean;
}

interface AppState {
  modals: ModalState;
  ui: {
    isLoading: boolean;
    error: string | null;
  };
}

const state = $state<AppState>({
  modals: {
    createExperiment: false,
    editExperiment: null,
    deleteExperiment: null,
    selectedExperiment: null,
    createWorkspace: false,
  },
  ui: {
    isLoading: false,
    error: null,
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

export function getModalState() {
  return state.modals;
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

export function setLoading(loading: boolean) {
  state.ui.isLoading = loading;
}

export function setError(error: string | null) {
  state.ui.error = error;
}

export function getUIState() {
  return state.ui;
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

export function clearAllModals() {
  state.modals.createExperiment = false;
  state.modals.editExperiment = null;
  state.modals.deleteExperiment = null;
  state.modals.selectedExperiment = null;
  state.modals.createWorkspace = false;
}
