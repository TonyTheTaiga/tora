import { PostgrestError, type SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "../database.types";
import type { Experiment, HyperParam, Workspace } from "$lib/types";

export function handleError(
  error: PostgrestError | null,
  context: string,
): void {
  if (error) {
    throw new Error(`${context}: ${error.message}`);
  }
}

export function mapToExperiment(data: any): Experiment {
  return {
    id: data.id,
    name: data.name,
    description: data.description ?? "",
    hyperparams: (data.hyperparams as HyperParam[]) ?? [],
    createdAt: new Date(data.created_at),
    updatedAt: new Date(data.updated_at),
    tags: data.tags ?? [],
    availableMetrics: data.availableMetrics,
  };
}

export function mapRpcResultToExperiment(row: any): Experiment {
  return {
    id: row.experiment_id,
    name: row.experiment_name,
    description: row.experiment_description ?? "",
    hyperparams: (row.experiment_hyperparams as HyperParam[]) ?? [],
    tags: row.experiment_tags ?? [],
    createdAt: new Date(row.experiment_created_at),
    updatedAt: new Date(row.experiment_updated_at),
    availableMetrics: row.available_metrics,
  };
}

export function mapToWorkspace(data: any): Workspace {
  return {
    id: data.id,
    name: data.name,
    description: data.description ? data.description : "",
    createdAt: new Date(data.created_at),
    role: data.user_workspaces[0].workspace_role.name,
  };
}

export abstract class BaseRepository {
  protected client: SupabaseClient<Database>;

  constructor(client: SupabaseClient<Database>) {
    this.client = client;
  }
}
