import { browser } from "$app/environment";
import type { Experiment } from "$lib/types";

const EXPERIMENTS_CACHE_VERSION = 1;
const EXPERIMENTS_CACHE_PREFIX = "experiments:";
export const DEFAULT_TTL_MS = 5 * 60 * 1000; // 5 minutes

type PersistedExperiment = Omit<Experiment, "createdAt" | "updatedAt"> & {
  createdAt: string;
  updatedAt: string;
};

type ExperimentsCacheRecord = {
  v: number;
  ts: number; // stored timestamp (ms)
  data: PersistedExperiment[];
};

function keyFor(workspaceId: string) {
  return `${EXPERIMENTS_CACHE_PREFIX}${workspaceId}`;
}

function readRecord(workspaceId: string): ExperimentsCacheRecord | null {
  if (!browser) return null;
  try {
    const raw = localStorage.getItem(keyFor(workspaceId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ExperimentsCacheRecord;
    if (parsed.v !== EXPERIMENTS_CACHE_VERSION) return null;
    return parsed;
  } catch {
    return null;
  }
}

export function loadExperimentsFromStorage(
  workspaceId: string,
  ttlMs: number = DEFAULT_TTL_MS,
): Experiment[] | null {
  const record = readRecord(workspaceId);
  if (!record) return null;
  if (Date.now() - record.ts > ttlMs) return null;
  return record.data.map((e) => ({
    ...e,
    createdAt: new Date(e.createdAt),
    updatedAt: new Date(e.updatedAt),
  }));
}

export function getExperimentsTimestamp(workspaceId: string): number | null {
  return readRecord(workspaceId)?.ts ?? null;
}

export function saveExperimentsToStorage(
  workspaceId: string,
  experiments: Experiment[],
): void {
  if (!browser) return;
  try {
    const record: ExperimentsCacheRecord = {
      v: EXPERIMENTS_CACHE_VERSION,
      ts: Date.now(),
      data: experiments.map((e) => ({
        ...e,
        createdAt: e.createdAt.toISOString(),
        updatedAt: e.updatedAt.toISOString(),
      })),
    };
    localStorage.setItem(keyFor(workspaceId), JSON.stringify(record));
  } catch {
    // ignore quota or serialization errors
  }
}

export function clearExperimentsFromStorage(workspaceId: string): void {
  if (!browser) return;
  try {
    localStorage.removeItem(keyFor(workspaceId));
  } catch {
    // ignore
  }
}

export function clearAllExperimentsFromStorage(): void {
  if (!browser) return;
  try {
    // remove all keys with prefix
    const keys: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(EXPERIMENTS_CACHE_PREFIX)) keys.push(k);
    }
    keys.forEach((k) => localStorage.removeItem(k));
  } catch {
    // ignore
  }
}
