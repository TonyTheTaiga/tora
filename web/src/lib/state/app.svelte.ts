import type { Experiment } from "$lib/types";
import { startTimer } from "$lib/utils/timing";

interface ModalState {
  createExperiment: boolean;
  editExperiment: Experiment | null;
  deleteExperiment: Experiment | null;
  selectedExperiment: Experiment | null;
}

interface AppState {
  modals: ModalState;
  ui: {
    isLoading: boolean;
    error: string | null;
  };
}

let state = $state<AppState>({
  modals: {
    createExperiment: false,
    editExperiment: null,
    deleteExperiment: null,
    selectedExperiment: null,
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

export function openEditExperimentModal(experiment: Experiment) {
  const timer = startTimer("ui.openEditModal", { experimentId: experiment.id });
  state.modals.editExperiment = experiment;
  timer.end();
}

export function closeEditExperimentModal() {
  state.modals.editExperiment = null;
}

export function openDeleteExperimentModal(experiment: Experiment) {
  const timer = startTimer("ui.openDeleteModal", { experimentId: experiment.id });
  state.modals.deleteExperiment = experiment;
  timer.end();
}

export function closeDeleteExperimentModal() {
  state.modals.deleteExperiment = null;
}

export function setSelectedExperiment(experiment: Experiment | null) {
  const timer = startTimer("ui.setSelectedExperiment", { 
    experimentId: experiment?.id,
    action: experiment ? "select" : "deselect"
  });
  
  state.modals.selectedExperiment = experiment;
  timer.end();
}

export function getModalState() {
  return state.modals;
}

export function getCreateExperimentModal() {
  return state.modals.createExperiment;
}

export function getEditExperimentModal() {
  return state.modals.editExperiment;
}

export function getDeleteExperimentModal() {
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

export function clearAllModals() {
  state.modals.createExperiment = false;
  state.modals.editExperiment = null;
  state.modals.deleteExperiment = null;
  state.modals.selectedExperiment = null;
}
