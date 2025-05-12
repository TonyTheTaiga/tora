import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database, Json } from "./database.types";
import type {
  Experiment,
  ExperimentAndMetrics,
  HyperParam,
  Metric,
} from "$lib/types";

export class DatabaseClient {
  private static instance: SupabaseClient<Database>;

  private static getInstance(): SupabaseClient<Database> {
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
  ): Promise<Experiment> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .insert({
        name,
        description,
        hyperparams: hyperparams as unknown as Json[],
        tags,
      })
      .select()
      .single();

    if (error || !data) {
      throw new Error(`Failed to create experiment: ${error?.message}`);
    }

    await DatabaseClient.getInstance().from("user_experiments").insert({ user_id: userId, experiment_id: data.id, role: "OWNER" }).select();

    return {
      id: data.id,
      name: data.name,
      description: data.description,
      hyperparams: data.hyperparams as unknown as HyperParam[],
      createdAt: new Date(data.created_at),
      tags: data.tags,
    };
  }

  static async getExperiments(query: string | null): Promise<Experiment[]> {
    if (!query) {
      query = "";
    }

    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (name)")
      .ilike("name", `%${query}%`)
      .order("created_at", { ascending: false });

    if (error) {
      throw new Error(`Failed to get experiments: ${error.message}`);
    }

    const result = data.map(
      (exp): Experiment => ({
        id: exp.id,
        name: exp.name,
        description: exp.description,
        hyperparams: exp.hyperparams as unknown as HyperParam[],
        createdAt: new Date(exp.created_at),
        tags: exp.tags,
        availableMetrics: [...new Set(exp.metric.map((m) => m.name))],
      }),
    );

    return result;
  }

  static async getExperiment(id: string): Promise<Experiment> {
    const { data, error } = await DatabaseClient.getInstance()
      .from("experiment")
      .select("*, metric (name)")
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
      availableMetrics: [...new Set(data.metric.map((m) => m.name))],
      tags: data.tags,
    };
  }

  static async getExperimentAndMetrics(
    id: string,
  ): Promise<ExperimentAndMetrics> {
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
        availableMetrics: [...new Set(data.metric.map((m) => m.name))],
        tags: data.tags,
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

