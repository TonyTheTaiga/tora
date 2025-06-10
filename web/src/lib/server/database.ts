import type { PostgrestError, SupabaseClient } from "@supabase/supabase-js";
import type { Database, Json } from "./database.types";
import type {
  Experiment,
  ExperimentAndMetrics,
  HyperParam,
  Metric,
  Visibility,
  Workspace,
} from "$lib/types";
import { timeAsync, startTimer } from "$lib/utils/timing";

function handleError(error: PostgrestError | null, context: string): void {
  if (error) {
    throw new Error(`${context}: ${error.message}`);
  }
}

function mapToExperiment(data: any, userIdOverride?: string): Experiment {
  const finalUserId =
    userIdOverride || data.user_experiments?.[0]?.user_id || "";
  if (!finalUserId && data.visibility !== "PUBLIC") {
    console.warn(`[DB Client] Experiment ${data.id} is missing a user_id.`);
  }

  return {
    id: data.id,
    user_id: finalUserId,
    name: data.name,
    description: data.description ?? "",
    hyperparams: (data.hyperparams as HyperParam[]) ?? [],
    createdAt: new Date(data.created_at),
    updatedAt: new Date(data.updated_at),
    tags: data.tags ?? [],
    visibility: data.visibility,
    availableMetrics: data.availableMetrics,
  };
}

function mapRpcResultToExperiment(row: any): Experiment {
  return {
    id: row.experiment_id,
    user_id: row.experiment_user_id,
    name: row.experiment_name,
    description: row.experiment_description ?? "",
    hyperparams: (row.experiment_hyperparams as HyperParam[]) ?? [],
    tags: row.experiment_tags ?? [],
    createdAt: new Date(row.experiment_created_at),
    updatedAt: new Date(row.experiment_updated_at),
    visibility: row.experiment_visibility,
    availableMetrics: row.available_metrics,
  };
}

function mapToWorkspace(data: any): Workspace {
  return {
    id: data.id,
    user_id: data.user_id,
    name: data.name,
    description: data.description,
    created_at: new Date(data.created_at),
  };
}

export function createDbClient(client: SupabaseClient<Database>) {
  return {
    // --- Experiment Methods ---

    async createExperiment(
      userId: string,
      details: {
        name: string;
        description: string;
        hyperparams: HyperParam[];
        tags: string[];
        visibility?: Visibility;
        workspaceId?: string;
      },
    ): Promise<Experiment> {
      const { data: expData, error: expError } = await client
        .from("experiment")
        .insert({
          name: details.name,
          description: details.description,
          hyperparams: details.hyperparams as unknown as Json[],
          tags: details.tags,
          visibility: details.visibility ?? "PRIVATE",
        })
        .select()
        .single();

      handleError(expError, "Failed to create experiment");
      if (!expData) throw new Error("Experiment creation returned no data.");

      const { error: userExpError } = await client
        .from("user_experiments")
        .insert({
          user_id: userId,
          experiment_id: expData.id,
          role: "OWNER",
        });
      handleError(userExpError, "Failed to link user to experiment");

      if (details.workspaceId) {
        const { error: wsExpError } = await client
          .from("workspace_experiments")
          .insert({
            workspace_id: details.workspaceId,
            experiment_id: expData.id,
          });
        handleError(wsExpError, "Failed to link workspace to experiment");
      }

      return mapToExperiment(expData, userId);
    },

    async getExperiments(
      userId: string,
      workspaceId?: string,
    ): Promise<Experiment[]> {
      return timeAsync(
        "db.getExperiments",
        async () => {
          const { data, error } = await client.rpc("get_user_experiments", {
            user_id: userId,
            workspace_id: workspaceId,
          });

          handleError(error, "Failed to get experiments");
          return data?.map(mapRpcResultToExperiment) ?? [];
        },
        { userId, workspaceId },
      );
    },

    async getExperiment(id: string): Promise<Experiment> {
      const { data, error } = await client
        .from("experiment")
        .select("*, user_experiments(user_id)")
        .eq("id", id)
        .single();

      handleError(error, `Failed to get experiment with ID ${id}`);
      if (!data) throw new Error(`Experiment with ID ${id} not found.`);
      return mapToExperiment(data);
    },

    async getExperimentAndMetrics(id: string): Promise<ExperimentAndMetrics> {
      return timeAsync(
        "db.getExperimentAndMetrics",
        async () => {
          const { data, error } = await client
            .from("experiment")
            .select("*, metric(*), user_experiments!inner(user_id)")
            .eq("id", id)
            .single();

          handleError(
            error,
            `Failed to get experiment and metrics for ID ${id}`,
          );
          if (!data) throw new Error(`Experiment with ID ${id} not found.`);

          return {
            experiment: mapToExperiment(data),
            metrics: (data.metric as Metric[]) ?? [],
          };
        },
        { experimentId: id },
      );
    },

    async checkExperimentAccess(id: string, userId?: string): Promise<void> {
      const { data, error } = await client
        .from("experiment")
        .select("visibility, user_experiments(user_id)")
        .eq("id", id)
        .single();

      handleError(error, `Failed to check access for experiment ID ${id}`);
      if (!data) throw new Error(`Experiment with ID ${id} not found.`);

      if (data.visibility === "PUBLIC") {
        return;
      }

      if (
        !userId ||
        !data.user_experiments.some((ue) => ue.user_id === userId)
      ) {
        throw new Error(`Access denied to experiment with ID ${id}`);
      }
    },

    async updateExperiment(
      id: string,
      update: Partial<Experiment>,
    ): Promise<void> {
      const { error } = await client
        .from("experiment")
        .update(update as any)
        .eq("id", id);
      handleError(error, `Failed to update experiment with ID ${id}`);
    },

    async deleteExperiment(id: string): Promise<void> {
      const { error } = await client.from("experiment").delete().eq("id", id);
      handleError(error, `Failed to delete experiment with ID ${id}`);
    },

    // --- Metric Methods ---

    async getMetrics(experimentId: string): Promise<Metric[]> {
      return timeAsync(
        "db.getMetrics",
        async () => {
          const { data, error } = await client
            .from("metric")
            .select("*")
            .eq("experiment_id", experimentId)
            .order("created_at", { ascending: false });

          handleError(
            error,
            `Failed to get metrics for experiment ${experimentId}`,
          );
          return (data as Metric[]) ?? [];
        },
        { experimentId },
      );
    },

    async createMetric(metric: Metric): Promise<void> {
      const { error } = await client.from("metric").insert(metric);
      handleError(error, "Failed to write metric");
    },

    async batchCreateMetric(metrics: Metric[]): Promise<void> {
      const { error } = await client.from("metric").insert(metrics);
      // For production apps, consider a more robust retry mechanism (e.g., using p-retry)
      handleError(error, "Failed to batch write metrics");
    },

    // --- Reference Methods ---

    async createReference(
      fromExperiment: string,
      toExperiment: string,
    ): Promise<void> {
      const { error } = await client.from("experiment_references").insert({
        from_experiment: fromExperiment,
        to_experiment: toExperiment,
      });
      handleError(error, "Failed to create experiment reference");
    },

    async deleteReference(
      fromExperiment: string,
      toExperiment: string,
    ): Promise<void> {
      const { error } = await client
        .from("experiment_references")
        .delete()
        .match({
          from_experiment: fromExperiment,
          to_experiment: toExperiment,
        });
      handleError(error, "Failed to delete experiment reference");
    },

    async getReferenceChain(experimentId: string): Promise<Experiment[]> {
      return timeAsync(
        "db.getReferenceChain",
        async () => {
          const { data, error } = await client.rpc("get_experiment_chain", {
            target_experiment_id: experimentId,
          });

          console.error(error);

          handleError(
            error,
            `Failed to get reference chain for experiment ${experimentId}`,
          );
          return (
            data?.map((item) => ({
              id: item.experiment_id,
              user_id: "",
              name: item.experiment_name,
              description: item.experiment_description ?? "",
              hyperparams:
                (item.experiment_hyperparams as unknown as HyperParam[]) ?? [],
              createdAt: new Date(item.experiment_created_at),
              updatedAt: new Date(item.experiment_updated_at),
              tags: item.experiment_tags ?? [],
              visibility: item.experiment_visibility,
              availableMetrics: [],
            })) ?? []
          );
        },
        { experimentId },
      );
    },

    // --- Workspace Methods ---

    async getWorkspaces(userId: string): Promise<Workspace[]> {
      const { data, error } = await client
        .from("workspace")
        .select("*")
        .eq("user_id", userId);

      handleError(error, "Failed to get workspaces");
      return data?.map(mapToWorkspace) ?? [];
    },

    async createWorkspace(
      name: string,
      description: string | null,
      userId: string,
    ): Promise<Workspace> {
      const { data, error } = await client
        .from("workspace")
        .insert({ name, description, user_id: userId })
        .select()
        .single();

      handleError(error, "Failed to create workspace");
      if (!data) throw new Error("Workspace creation returned no data.");
      return mapToWorkspace(data);
    },

    async getOrCreateDefaultWorkspace(userId: string): Promise<Workspace> {
      const workspaces = await this.getWorkspaces(userId);
      if (workspaces[0]) {
        return workspaces[0];
      }

      return this.createWorkspace("Default", "Your default workspace", userId);
    },

    // --- API Key Methods ---

    async getApiKeys(userId: string) {
      const { data, error } = await client
        .from("api_keys")
        .select("id, name, created_at, last_used, revoked")
        .eq("user_id", userId)
        .eq("revoked", false)
        .order("created_at", { ascending: false });

      handleError(error, "Failed to get API keys");
      return data ?? [];
    },

    async createApiKey(userId: string, name: string, keyHash: string) {
      const { data, error } = await client
        .from("api_keys")
        .insert({
          key_hash: keyHash,
          name,
          user_id: userId,
          revoked: false,
        })
        .select()
        .single();

      handleError(error, "Failed to create API key");
      if (!data) throw new Error("API key creation returned no data.");
      return data;
    },

    async revokeApiKey(userId: string, keyId: string): Promise<void> {
      const { data: keyData, error: fetchError } = await client
        .from("api_keys")
        .select("id")
        .eq("id", keyId)
        .eq("user_id", userId)
        .single();

      if (fetchError || !keyData) {
        throw new Error("API key not found");
      }

      const { error: updateError } = await client
        .from("api_keys")
        .update({ revoked: true })
        .eq("id", keyId)
        .eq("user_id", userId);

      handleError(updateError, "Failed to revoke API key");
    },

    async lookupApiKey(keyHash: string): Promise<{ user_id: string } | null> {
      const { data, error } = await client
        .from("api_keys")
        .select("user_id")
        .eq("key_hash", keyHash)
        .eq("revoked", false)
        .single();

      if (error || !data?.user_id) {
        if (error) console.error("API key lookup error:", error.message);
        return null;
      }

      return { user_id: data.user_id };
    },

    async updateApiKeyLastUsed(keyHash: string): Promise<void> {
      const { error } = await client
        .from("api_keys")
        .update({ last_used: new Date().toISOString() })
        .eq("key_hash", keyHash)
        .eq("revoked", false);

      if (error) {
        console.warn("Failed to update API key last_used:", error.message);
      }
    },

    async getExperimentsAndMetrics(experimentIds: string[]): Promise<any[]> {
      return timeAsync(
        "db.getExperimentsAndMetrics",
        async () => {
          const { data, error } = await client.rpc(
            "get_experiments_and_metrics",
            {
              experiment_ids: experimentIds,
            },
          );

          handleError(error, "Failed to get experiments and metrics");
          return data ?? [];
        },
        { experimentCount: experimentIds.length },
      );
    },
  };
}
