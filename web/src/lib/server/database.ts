import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database, Json } from "./database.types";
import type {
  Experiment,
  ExperimentAndMetrics,
  HyperParam,
  Metric,
  Visibility,
  Workspace,
} from "$lib/types";

export class DatabaseClient {
  private static instance: SupabaseClient<Database>;

  static getInstance(): SupabaseClient<Database> {
    if (!this.instance) {
      throw new Error("DB client not set. Did you forget to set it?");
    }
    return this.instance;
  }

  static setInstance(instance: SupabaseClient<Database>) {
    this.instance = instance;
  }

  static async createExperiment(
    userId: string,
    name: string,
    description: string,
    hyperparams: HyperParam[],
    tags: string[],
    visibility: Visibility = "PRIVATE",
    workspaceId?: string,
  ): Promise<Experiment> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .insert({
        name,
        description,
        hyperparams: hyperparams as unknown as Json[],
        tags,
        visibility,
      })
      .select()
      .single();

    if (error || !data) {
      throw new Error(`Failed to create experiment: ${error?.message}`);
    }

    await DatabaseClient.getInstance()
      .from("user_experiments")
      .insert({ user_id: userId, experiment_id: data.id, role: "OWNER" })
      .select();

    // Link experiment to workspace if provided
    if (workspaceId) {
      await DatabaseClient.getInstance()
        .from("workspace_experiments")
        .insert({ workspace_id: workspaceId, experiment_id: data.id });
    }

    return {
      id: data.id,
      user_id: userId,
      name: data.name,
      description: data.description,
      hyperparams: hyperparams,
      availableMetrics: [],
      createdAt: new Date(data.created_at),
      tags: data.tags,
      visibility: data.visibility,
    };
  }

  static async getExperiments(
    query: string | null,
    userId?: string,
    workspaceId?: string,
  ): Promise<Experiment[]> {
    if (!query) {
      query = "";
    }

    let data: any[] = [];
    let error: any = null;

    if (!userId) {
      const result = await DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments (user_id)")
        .ilike("name", `%${query}%`)
        .eq("visibility", "PUBLIC")
        .order("created_at", { ascending: false });

      data = result.data || [];
      error = result.error;
    } else if (workspaceId) {
      // Get experiments for a specific workspace
      const result = await DatabaseClient.getInstance()
        .from("experiment")
        .select(
          "*, metric (name), user_experiments (user_id), workspace_experiments!inner (workspace_id)",
        )
        .ilike("name", `%${query}%`)
        .eq("workspace_experiments.workspace_id", workspaceId)
        .order("created_at", { ascending: false });

      data = result.data || [];
      error = result.error;
    } else {
      const publicExperimentsQuery = DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments (user_id)")
        .eq("visibility", "PUBLIC")
        .ilike("name", `%${query}%`)
        .order("created_at", { ascending: false });

      const userExperimentsQuery = DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments!inner (user_id, role)")
        .ilike("name", `%${query}%`)
        .eq("user_experiments.user_id", userId)
        .order("created_at", { ascending: false });

      const [publicResults, userResults] = await Promise.all([
        publicExperimentsQuery,
        userExperimentsQuery,
      ]);
      data = [];
      if (!publicResults.error && publicResults.data) {
        data = [...publicResults.data];
      }
      if (!userResults.error && userResults.data) {
        data = [...data, ...userResults.data];
      }
      error = publicResults.error || userResults.error;
    }
    if (error) {
      throw new Error(`Failed to get experiments: ${error.message}`);
    }
    const seenExperiments = new Set<string>();
    const result = data
      .sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      )
      .filter((exp) => {
        if (seenExperiments.has(exp.id)) return false;
        seenExperiments.add(exp.id);
        return true;
      })
      .map(
        (exp): Experiment => ({
          id: exp.id,
          user_id: exp.user_experiments[0].user_id,
          name: exp.name,
          description: exp.description,
          hyperparams: exp.hyperparams,
          createdAt: new Date(exp.created_at),
          tags: exp.tags,
          visibility: exp.visibility,
          availableMetrics: Array.from(
            new Set(
              ((exp.metric ?? []) as { name: string }[]).map((m) => m.name),
            ),
          ),
        }),
      );
    return result;
  }

  static async checkExperimentAccess(
    id: string,
    userId?: string,
  ): Promise<void> {
    if (!userId) {
      const { data: visibilityCheck, error: visibilityError } =
        await DatabaseClient.getInstance()
          .from("experiment")
          .select("visibility")
          .eq("id", id)
          .single();

      if (visibilityError || !visibilityCheck) {
        throw new Error(
          `Failed to get experiment with ID ${id}: ${visibilityError?.message}`,
        );
      }

      if (visibilityCheck.visibility !== "PUBLIC") {
        throw new Error(`Access denied to experiment with ID ${id}`);
      }
    } else {
      const { count, error: accessError } = await DatabaseClient.getInstance()
        .from("user_experiments")
        .select("*", { count: "exact", head: true })
        .eq("experiment_id", id)
        .eq("user_id", userId);

      const { data: visibilityCheck, error: visibilityError } =
        await DatabaseClient.getInstance()
          .from("experiment")
          .select("visibility")
          .eq("id", id)
          .single();

      if (visibilityError || !visibilityCheck) {
        throw new Error(
          `Failed to get experiment with ID ${id}: ${visibilityError?.message}`,
        );
      }

      if (visibilityCheck.visibility !== "PUBLIC" && (!count || count === 0)) {
        throw new Error(`Access denied to experiment with ID ${id}`);
      }
    }
  }

  static async getExperiment(id: string, userId?: string): Promise<Experiment> {
    // Not needed because getExperiments wont return invalid experimetns
    // await DatabaseClient.checkExperimentAccess(id, userId);

    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (name), user_experiments (user_id)")
      .eq("id", id)
      .single();

    if (error || !data) {
      throw new Error(
        `Failed to get experiment with ID ${id}: ${error?.message}`,
      );
    }

    return {
      id: data.id,
      user_id: userId,
      name: data.name,
      description: data.description,
      hyperparams: data.hyperparams as unknown as HyperParam[],
      createdAt: new Date(data.created_at),
      availableMetrics: [...new Set((data.metric || []).map((m) => m.name))],
      tags: data.tags,
      visibility: data.visibility,
    };
  }

  static async getExperimentAndMetrics(
    id: string,
    userId?: string,
  ): Promise<ExperimentAndMetrics> {
    // Not needed because getExperiments wont return invalid experimetns
    // await DatabaseClient.checkExperimentAccess(id, userId);

    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (*)")
      .eq("id", id)
      .single();

    if (error || !data) {
      throw new Error(
        `Failed to get experiment with ID ${id}: ${error?.message}`,
      );
    }

    return {
      experiment: {
        id: data.id,
        user_id: userId,
        name: data.name,
        description: data.description,
        hyperparams: data.hyperparams as unknown as HyperParam[],
        createdAt: new Date(data.created_at),
        availableMetrics: [...new Set((data.metric || []).map((m) => m.name))],
        tags: data.tags,
        visibility: data.visibility,
      },
      metrics: data.metric as Metric[],
    };
  }

  static async deleteExperiment(id: string): Promise<void> {
    const { error } = await DatabaseClient.getInstance()
      .from("experiment")
      .delete()
      .eq("id", id);

    if (error) {
      throw new Error(
        `Failed to delete experiment with ID ${id}: ${error.message}`,
      );
    }
  }

  static async updateExperiment(
    id: string,
    update: { [key: string]: any },
  ): Promise<void> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .update(update)
      .eq("id", id)
      .select();

    if (error) {
      throw new Error(
        `Failed to update experiment with ID ${id}: ${error.message}`,
      );
    }
  }

  static async getMetrics(experimentId: string): Promise<Metric[]> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("metric")
      .select()
      .eq("experiment_id", experimentId)
      .order("created_at", { ascending: false });

    if (error) {
      throw new Error(`Failed to get metrics: ${error.message}`);
    }

    return data as Metric[];
  }

  static async createMetric(metric: Metric): Promise<void> {
    const { error } = await DatabaseClient.getInstance()
      .from("metric")
      .insert(metric);

    if (error) {
      throw new Error(`Failed to write metric: ${error.message}`);
    }
  }

  static async batchCreateMetric(metrics: Metric[]): Promise<void> {
    const maxRetries = 3;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        const { error } = await DatabaseClient.getInstance()
          .from("metric")
          .insert(metrics);

        if (!error) return;
        lastError = new Error(`Batch insert failed: ${error.message}`);
      } catch (error) {
        lastError =
          error instanceof Error ? error : new Error("Unknown error occurred");

        if (attempt < maxRetries - 1) {
          await new Promise((resolve) =>
            setTimeout(resolve, 1000 * Math.pow(2, attempt)),
          );
          continue;
        }
      }
    }

    throw new Error(
      `Failed to write metrics after ${maxRetries} retries: ${lastError?.message}`,
    );
  }

  static async createReference(fromExperiment: string, toExperiment: string) {
    const { error } = await DatabaseClient.getInstance()
      .from("experiment_references")
      .insert({
        from_experiment: fromExperiment,
        to_experiment: toExperiment,
      });

    if (error) {
      throw new Error(`Failed to create reference: ${error.message}`);
    }
  }

  static async deleteReference(fromExperiment: string, toExperiment: string) {
    const { error } = await DatabaseClient.getInstance()
      .from("experiment_references")
      .delete()
      .match({
        from_experiment: fromExperiment,
        to_experiment: toExperiment,
      });

    if (error) {
      throw new Error(`Failed to delete reference: ${error.message}`);
    }
  }

  static async getReferenceChain(
    experimentUuid: string,
  ): Promise<Experiment[]> {
    const { data, error } = await DatabaseClient.getInstance().rpc(
      "get_experiment_chain",
      { target_experiment_id: experimentUuid },
    );
    if (error) {
      throw new Error(`Failed to get references: ${error.message}`);
    }
    return data.map((item) => ({
      id: item.id,
      user_id: undefined,
      name: item.name,
      description: item.description,
      hyperparams: item.hyperparams as unknown as HyperParam[],
      availableMetrics: [],
      tags: item.tags,
      createdAt: new Date(item.created_at),
      visibility: item.visibility,
    }));
  }

  static async getWorkspaces(userId?: string): Promise<Workspace[]> {
    let query = DatabaseClient.getInstance().from("workspace").select("*");

    if (userId) {
      query = query.eq("user_id", userId);
    }

    const { data, error } = await query;

    if (error) {
      throw new Error(`Failed to get workspaces: ${error.message}`);
    }

    return data.map((item) => ({
      id: item.id,
      user_id: item.user_id,
      name: item.name,
      description: item.description,
      created_at: new Date(item.created_at),
    }));
  }

  static async createWorkspace(
    name: string,
    description: string | null,
    user_id: string,
  ): Promise<Workspace> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("workspace")
      .insert({
        name: name,
        description: description,
        user_id: user_id,
      })
      .select()
      .single();

    if (error) {
      throw new Error(`Failed to create workspace: ${error.message}`);
    }

    return {
      id: data.id,
      user_id: data.user_id,
      name: data.name,
      description: data.description,
      created_at: new Date(data.created_at),
    };
  }

  static async getOrCreateDefaultWorkspace(userId: string): Promise<Workspace> {
    const workspaces = await DatabaseClient.getWorkspaces(userId);

    if (workspaces.length > 0) {
      return workspaces[0];
    }

    return await DatabaseClient.createWorkspace(
      "Default",
      "Your default workspace for experiments",
      userId,
    );
  }
}

export const {
  createExperiment,
  getExperiments,
  getExperiment,
  getExperimentAndMetrics,
  deleteExperiment,
  updateExperiment,
  getMetrics,
  createReference,
  deleteReference,
  getReferenceChain,
  createMetric,
  batchCreateMetric,
  getWorkspaces,
  createWorkspace,
  getOrCreateDefaultWorkspace,
} = DatabaseClient;
