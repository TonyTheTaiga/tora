import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database, Json } from "./database.types";
import type {
  Experiment,
  ExperimentAndMetrics,
  HyperParam,
  Metric,
  Visibility,
} from "$lib/types";

export class DatabaseClient {
  private static instance: SupabaseClient<Database>;

  static getInstance(): SupabaseClient<Database> {
    if (!this.instance) {
      throw new Error("DB client not set. Did you forget to set it?")
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

    await DatabaseClient.getInstance().from("user_experiments").insert({ user_id: userId, experiment_id: data.id, role: "OWNER" }).select();


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

  static async getExperiments(query: string | null, userId?: string): Promise<Experiment[]> {
    if (!query) {
      query = "";
    }

    let data: any[] = [];
    let error: any = null;

    // Anonymous users can only see PUBLIC experiments
    if (!userId) {
      const result = await DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments (user_id)")
        .ilike("name", `%${query}%`)
        .eq("visibility", "PUBLIC")
        .order("created_at", { ascending: false });

      data = result.data || [];
      error = result.error;
    } else {
      // Logged-in users can see:
      // 1. Public experiments
      // 2. Their own experiments (OWNER, COLLABORATOR)

      // 1. Get public experiments
      const publicExperimentsQuery = DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments (user_id)")
        .eq("visibility", "PUBLIC")
        .ilike("name", `%${query}%`)
        .order("created_at", { ascending: false });

      // 2. Get experiments where user is a collaborator or owner
      const userExperimentsQuery = DatabaseClient.getInstance()
        .from("experiment")
        .select("*, metric (name), user_experiments!inner (user_id, role)")
        .ilike("name", `%${query}%`)
        .eq("user_experiments.user_id", userId)
        .order("created_at", { ascending: false });

      // 3. Run both queries
      const [publicResults, userResults] = await Promise.all([
        publicExperimentsQuery,
        userExperimentsQuery
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
      .sort((a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      )
      .filter(exp => {
        if (seenExperiments.has(exp.id)) return false;
        seenExperiments.add(exp.id);
        return true;
      })
      .map((exp): Experiment => ({
        id: exp.id,
        user_id: exp.user_experiments[0].user_id,
        name: exp.name,
        description: exp.description,
        hyperparams: exp.hyperparams as unknown as HyperParam[],
        createdAt: new Date(exp.created_at),
        tags: exp.tags,
        visibility: exp.visibility,
        availableMetrics: Array.from(
          new Set(
            ((exp.metric ?? []) as { name: string }[]).map(
              (m) => m.name
            )
          )
        ),
      }));
    return result;
  }

  /**
   * Check if a user has access to an experiment.
   * Anonymous users can only access PUBLIC experiments.
   * Logged-in users can access their own experiments and PUBLIC experiments.
   */
  static async checkExperimentAccess(id: string, userId?: string): Promise<void> {
    // Anonymous users can only access PUBLIC experiments
    if (!userId) {
      const { data: visibilityCheck, error: visibilityError } = await DatabaseClient.getInstance()
        .from("experiment")
        .select("visibility")
        .eq("id", id)
        .single();

      if (visibilityError || !visibilityCheck) {
        throw new Error(`Failed to get experiment with ID ${id}: ${visibilityError?.message}`);
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

      const { data: visibilityCheck, error: visibilityError } = await DatabaseClient.getInstance()
        .from("experiment")
        .select("visibility")
        .eq("id", id)
        .single();

      if (visibilityError || !visibilityCheck) {
        throw new Error(`Failed to get experiment with ID ${id}: ${visibilityError?.message}`);
      }

      if (visibilityCheck.visibility !== "PUBLIC" && (!count || count === 0)) {
        throw new Error(`Access denied to experiment with ID ${id}`);
      }
    }
  }

  static async getExperiment(id: string, userId?: string): Promise<Experiment> {
    // First check access permissions
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
    // First check access permissions
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
      name: item.name,
      description: item.description,
      hyperparams: item.hyperparams as unknown as HyperParam[],
      tags: item.tags,
      createdAt: new Date(item.created_at),
      visibility: item.visibility,
    }));
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
  createMetric,
  batchCreateMetric,
  createReference,
  getReferenceChain,
} = DatabaseClient;
