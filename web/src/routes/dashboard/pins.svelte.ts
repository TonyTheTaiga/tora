// Simple per-experiment pinned names store with localStorage persistence
let pins = $state(new Map<string, string[]>());

function storageKey(expId: string) {
  return `tora:pinnedResults:${expId}`;
}

function readFromStorage(expId: string): string[] {
  try {
    if (typeof localStorage === "undefined") return [];
    const raw = localStorage.getItem(storageKey(expId));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((x: unknown): x is string => typeof x === "string")
      : [];
  } catch {
    return [];
  }
}

function writeToStorage(expId: string, names: string[]) {
  try {
    if (typeof localStorage === "undefined") return;
    localStorage.setItem(storageKey(expId), JSON.stringify(names));
  } catch {}
}

export function loadPins(expId: string): string[] {
  const names = readFromStorage(expId);
  pins.set(expId, names);
  return names;
}

export function getPins(expId: string): string[] {
  return pins.get(expId) ?? [];
}

export function isPinned(expId: string, name: string): boolean {
  return getPins(expId).includes(name);
}

export function addPin(expId: string, name: string) {
  const current = getPins(expId);
  if (current.includes(name)) return;
  const next = [...current, name];
  pins.set(expId, next);
  writeToStorage(expId, next);
}

export function removePin(expId: string, name: string) {
  const current = getPins(expId);
  if (!current.includes(name)) return;
  const next = current.filter((n) => n !== name);
  pins.set(expId, next);
  writeToStorage(expId, next);
}

export function togglePin(expId: string, name: string) {
  if (isPinned(expId, name)) removePin(expId, name);
  else addPin(expId, name);
}

export function mapPinnedResults<T extends { name?: string }>(
  expId: string,
  results: T[],
): T[] {
  const names = new Set(getPins(expId));
  return results.filter((r) => (r?.name ? names.has(r.name) : false));
}
