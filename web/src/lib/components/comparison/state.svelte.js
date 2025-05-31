let comparisonMode = $state(false);

export function getMode() {
  return comparisonMode;
}

export function toggleMode() {
  comparisonMode = !comparisonMode;
}
