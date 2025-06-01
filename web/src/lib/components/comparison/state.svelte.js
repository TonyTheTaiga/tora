let state = $state({
  comparisonMode: false,
  comparisonExperimentIds: [],
});

export function getMode() {
  return state.comparisonMode;
}

export function toggleMode() {
  state.comparisonMode = !state.comparisonMode;
  if (state.comparisonMode === false) {
    // do stuff on exit
    state.comparisonExperimentIds = [];
  }
}

export function getExperimentsSelectedForComparision() {
  return state.comparisonExperimentIds;
}

export function addExperiment(experimentId) {
  console.log($state.snapshot(state.comparisonExperimentIds));
  if (state.comparisonExperimentIds.includes(experimentId)) {
    state.comparisonExperimentIds = state.comparisonExperimentIds.filter(
      (item) => item !== experimentId,
    );
  } else {
    state.comparisonExperimentIds.push(experimentId);
  }
}

export function selectedForComparison(experimentId) {
  return state.comparisonExperimentIds.includes(experimentId);
}
