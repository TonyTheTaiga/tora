interface ComparisonState {
  comparisonMode: boolean;
  comparisonIds: string[];
}

let state = $state<ComparisonState>({
  comparisonMode: false,
  comparisonIds: [],
});

export function getMode() {
  return state.comparisonMode;
}

export function toggleMode() {
  state.comparisonMode = !state.comparisonMode;
  if (state.comparisonMode === false) {
    // do stuff on exit
    state.comparisonIds = [];
  }
}

export function getExperimentsSelectedForComparision() {
  return state.comparisonIds;
}

export function addExperiment(id: string) {
  if (state.comparisonIds.includes(id)) {
    state.comparisonIds = state.comparisonIds.filter((item) => item !== id);
  } else {
    state.comparisonIds.push(id);
  }
}

export function selectedForComparison(id: string) {
  return state.comparisonIds.includes(id);
}
