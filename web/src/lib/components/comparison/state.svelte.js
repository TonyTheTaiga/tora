let state = $state({
  comparisonMode: false,
  comparisonExperiments: [],
});

export function getMode() {
  return state.comparisonMode;
}

export function toggleMode() {
  state.comparisonMode = !state.comparisonMode;
  if (state.comparisonMode === false) {
    // do stuff on exit
    state.comparisonExperiments = [];
  }
}

export function getExperimentsSelectedForComparision() {
  return state.comparisonExperiments;
}

export function addExperiment(experiment) {
  if (state.comparisonExperiments.includes(experiment)) {
    state.comparisonExperiments = state.comparisonExperiments.filter(
      (item) => item !== experiment,
    );
  } else {
    state.comparisonExperiments.push(experiment);
  }
}

export function selectedForComparison(experiment) {
  return state.comparisonExperiments.includes(experiment);
}
