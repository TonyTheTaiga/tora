import { PostgrestError, type SupabaseClient } from "@supabase/supabase-js";
import type { Database, Json } from "./database.types";
import type {
  Experiment,
  ExperimentAndMetrics,
  HyperParam,
  Metric,
  Visibility,
  Workspace,
  ApiKey,
  PendingInvitation,
} from "$lib/types";
import { timeAsync } from "$lib/utils/timing";

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
    name: data.name,
    description: data.description ? data.description : "",
    createdAt: new Date(data.created_at),
    role: data.user_workspaces[0].workspace_role.name,
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
        workspaceId: string;
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
          creator: userId,
        })
        .select()
        .single();

      handleError(expError, "Failed to create experiment");
      if (!expData) throw new Error("Experiment creation returned no data.");

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

    async getExperiments(workspaceId: string): Promise<Experiment[]> {
      return timeAsync(
        "db.getExperiments",
        async () => {
          const { data, error } = await client.rpc("get_user_experiments", {
            workspace_id: workspaceId,
          });

          handleError(error, "Failed to get experiments");
          return data?.map(mapRpcResultToExperiment) ?? [];
        },
        { workspaceId },
      );
    },

    async getExperiment(id: string): Promise<Experiment> {
      const { data, error } = await client
        .from("experiment")
        .select("*")
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
            .select("*, metric(*)")
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
        .select(
          "visibility, creator, workspace_experiments(workspace_id, workspace(user_workspaces(user_id)))",
        )
        .eq("id", id)
        .single();

      handleError(error, `Failed to check access for experiment ID ${id}`);
      if (!data) throw new Error(`Experiment with ID ${id} not found.`);

      if (data.visibility === "PUBLIC") {
        return;
      }

      if (!userId) {
        throw new Error(`Access denied to experiment with ID ${id}`);
      }

      if (data.creator === userId) {
        return;
      }

      const hasWorkspaceAccess = data.workspace_experiments?.some((we) =>
        we.workspace?.user_workspaces?.some((uw) => uw.user_id === userId),
      );

      if (!hasWorkspaceAccess) {
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

    async getWorkspacesV2(
      userId: string,
      roles: string[],
    ): Promise<Workspace[]> {
      const { data, error } = await client
        .from("workspace")
        .select(
          `
          id,
          name,
          description,
          created_at,
          user_workspaces!inner (
            user_id,
            workspace_role (
              name
            )
          )
        `,
        )
        .eq("user_workspaces.user_id", userId)
        .in("user_workspaces.workspace_role.name", roles);

      handleError(error, "Failed to get workspaces");
      return data?.map(mapToWorkspace) ?? [];
    },

    async getWorkspacesAndExperiments(
      userId: string,
      roles: string[],
    ): Promise<{ workspaces: Workspace[]; experiments: Experiment[] }> {
      return timeAsync(
        "db.getWorkspacesAndExperiments",
        async () => {
          // Get workspaces for the user
          const { data: workspaceData, error: workspaceError } = await client
            .from("workspace")
            .select(
              `
              id,
              name,
              description,
              created_at,
              user_workspaces!inner (
                user_id,
                workspace_role (
                  name
                )
              )
            `,
            )
            .eq("user_workspaces.user_id", userId)
            .in("user_workspaces.workspace_role.name", roles);

          handleError(workspaceError, "Failed to get workspaces");
          const workspaces = workspaceData?.map(mapToWorkspace) ?? [];

          if (workspaces.length === 0) {
            return { workspaces: [], experiments: [] };
          }

          // Get all experiments for these workspaces in a single query
          const workspaceIds = workspaces.map(w => w.id);
          const { data: experimentData, error: experimentError } = await client
            .from("workspace_experiments")
            .select(
              `
              experiment:experiment_id (
                id,
                name,
                description,
                hyperparams,
                tags,
                created_at,
                updated_at,
                visibility
              )
            `,
            )
            .in("workspace_id", workspaceIds);

          handleError(experimentError, "Failed to get experiments");
          
          // Map and deduplicate experiments
          const experiments: Experiment[] = [];
          const seenExperimentIds = new Set<string>();
          
          experimentData?.forEach(item => {
            if (item.experiment && !seenExperimentIds.has(item.experiment.id)) {
              seenExperimentIds.add(item.experiment.id);
              experiments.push({
                id: item.experiment.id,
                name: item.experiment.name,
                description: item.experiment.description ?? "",
                hyperparams: (item.experiment.hyperparams as unknown as HyperParam[]) ?? [],
                tags: item.experiment.tags ?? [],
                createdAt: new Date(item.experiment.created_at),
                updatedAt: new Date(item.experiment.updated_at),
                visibility: item.experiment.visibility,
                availableMetrics: [], // Will be populated if needed
              });
            }
          });

          return { workspaces, experiments };
        },
        { userId },
      );
    },

    async createWorkspace(
      name: string,
      description: string | null,
      userId: string,
    ): Promise<Workspace> {
      const { data: workspaceData, error: workspaceError } = await client
        .from("workspace")
        .insert({ name, description })
        .select()
        .single();

      handleError(workspaceError, "Failed to create workspace");
      if (!workspaceData)
        throw new Error("Workspace creation returned no data.");

      const { data: ownerRole, error: roleError } = await client
        .from("workspace_role")
        .select("id")
        .eq("name", "OWNER")
        .single();

      handleError(roleError, "Failed to get OWNER role");
      if (!ownerRole) throw new Error("OWNER role not found");

      const { error: userWorkspaceError } = await client
        .from("user_workspaces")
        .insert({
          user_id: userId,
          workspace_id: workspaceData.id,
          role_id: ownerRole.id,
        });

      handleError(userWorkspaceError, "Failed to add user to workspace");

      const workspaceWithRole = {
        ...workspaceData,
        user_workspaces: [{ workspace_role: { name: "OWNER" } }],
      };

      return mapToWorkspace(workspaceWithRole);
    },

    async deleteWorkspace(id: string) {
      const { error } = await client.from("workspace").delete().eq("id", id);
      handleError(error, "Failed to delete workspace");
    },

    async removeWorkspaceRole(workspaceID: string, userId: string) {
      const { error } = await client
        .from("user_workspaces")
        .delete()
        .eq("user_id", userId)
        .eq("workspace_id", workspaceID);
      handleError(error, "Failed to remove workspace role");
    },

    // --- API Key Methods ---
    async getApiKeys(userId: string): Promise<ApiKey[]> {
      const { data, error } = await client
        .from("api_keys")
        .select("id, name, created_at, last_used, revoked")
        .eq("user_id", userId)
        .eq("revoked", false)
        .order("created_at", { ascending: false });

      handleError(error, "Failed to get API keys");
      return (
        data?.map((row) => ({
          id: row.id,
          name: row.name,
          createdAt: new Date(row.created_at),
          lastUsed: new Date(row.last_used),
          revoked: row.revoked,
        })) ?? []
      );
    },

    async createApiKey(
      userId: string,
      name: string,
      keyHash: string,
    ): Promise<ApiKey> {
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
      return {
        id: data.id,
        name: data.name,
        revoked: data.revoked,
        createdAt: new Date(data.created_at),
        lastUsed: new Date(data.last_used),
      };
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
        { experimentCount: experimentIds.length.toString() },
      );
    },

    // Workspace Invitations
    async createInvitation(
      from: string,
      to: string,
      workspace_id: string,
      role_id: string,
    ): Promise<PendingInvitation> {
      return timeAsync(
        "db.createInvitation",
        async () => {
          const { data, error } = await client
            .from("workspace_invitations")
            .insert({
              from: from,
              to: to,
              workspace_id: workspace_id,
              role_id: role_id,
              status: "PENDING",
            })
            .select()
            .single();

          if (error) {
            handleError(error, "Failed to create invitation");
          }
          if (!data) {
            throw new Error("unknown error");
          }

          return {
            id: data.id,
            from: data.from,
            to: data.to,
            workspaceId: data.workspace_id,
            roleId: data.role_id,
            status: data.status,
            createdAt: new Date(data.created_at),
          };
        },
        {
          from: from,
          to: to,
        },
      );
    },

    async markInvitationAsAccepted(id: string) {
      return timeAsync(
        "db.markInvitationMarked",
        async () => {
          const { error } = await client
            .from("workspace_invitations")
            .update({ status: "accepted" })
            .eq("id", id);

          if (error) {
            handleError(error, "failed to update invitation");
          }
        },
        { id },
      );
    },

    async getPendingInvitationsFrom(
      userId: string,
      status: string,
    ): Promise<PendingInvitation[]> {
      return timeAsync(
        "db.getPendingInvitationsFrom",
        async () => {
          const { data, error } = await client
            .from("workspace_invitations")
            .select("*")
            .eq("from", userId)
            .eq("status", status);

          handleError(error, "Failed to get pending invitations");
          if (!data) {
            return [];
          }

          return data.map((item) => ({
            id: item.id,
            from: item.from,
            to: item.to,
            workspaceId: item.workspace_id,
            roleId: item.role_id,
            createdAt: new Date(item.created_at),
            status: item.status,
          }));
        },
        { userId },
      );
    },
    async getPendingInvitationsTo(
      userId: string,
      status: string,
    ): Promise<PendingInvitation[]> {
      return timeAsync(
        "db.getPendingInvitationsTo",
        async () => {
          const { data, error } = await client
            .from("workspace_invitations")
            .select("*")
            .eq("to", userId)
            .eq("status", status);

          handleError(error, "Failed to get pending invitations");
          if (!data) {
            return [];
          }

          return data.map((item) => ({
            id: item.id,
            from: item.from,
            to: item.to,
            workspaceId: item.workspace_id,
            roleId: item.role_id,
            createdAt: new Date(item.created_at),
            status: item.status,
          }));
        },
        { userId },
      );
    },
  };
}
