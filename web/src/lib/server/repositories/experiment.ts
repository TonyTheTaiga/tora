import type { Json } from "../database.types";
import type { Experiment, HyperParam } from "$lib/types";
import { timeAsync } from "$lib/utils/timing";
import {
  BaseRepository,
  handleError,
  mapToExperiment,
  mapRpcResultToExperiment,
} from "./base";

export class ExperimentRepository extends BaseRepository {
  async createExperiment(
    userId: string,
    details: {
      name: string;
      description: string;
      hyperparams: HyperParam[];
      tags: string[];
    },
  ): Promise<Experiment> {
    const { data: expData, error: expError } = await this.client
      .from("experiment")
      .insert({
        name: details.name,
        description: details.description,
        hyperparams: details.hyperparams as unknown as Json[],
        tags: details.tags,
      })
      .select()
      .single();

    handleError(expError, "Failed to create experiment");
    if (!expData) throw new Error("Experiment creation returned no data.");
    return mapToExperiment(expData);
  }

  async getExperiments(workspaceId: string): Promise<Experiment[]> {
    return timeAsync(
      "db.getExperiments",
      async () => {
        const { data, error } = await this.client
          .from("experiment")
          .select(
            `
            id,
            created_at,
            updated_at,
            name,
            description,
            hyperparams,
            tags,
            workspace_experiments!inner(workspace_id),
            metric(name)
          `,
          )
          .eq("workspace_experiments.workspace_id", workspaceId)
          .order("created_at", { ascending: false });

        handleError(error, "Failed to get experiments");

        if (!data) return [];

        const experimentsWithMetrics = data.reduce((acc: any[], row: any) => {
          const existingExp = acc.find((exp) => exp.id === row.id);
          if (existingExp) {
            if (
              row.metric?.name &&
              !existingExp.available_metrics.includes(row.metric.name)
            ) {
              existingExp.available_metrics.push(row.metric.name);
            }
          } else {
            acc.push({
              experiment_id: row.id,
              experiment_created_at: row.created_at,
              experiment_updated_at: row.updated_at,
              experiment_name: row.name,
              experiment_description: row.description,
              experiment_hyperparams: row.hyperparams,
              experiment_tags: row.tags,
              available_metrics: row.metric?.name ? [row.metric.name] : [],
            });
          }
          return acc;
        }, []);

        return experimentsWithMetrics.map(mapRpcResultToExperiment);
      },
      { workspaceId },
    );
  }

  async getExperiment(id: string): Promise<Experiment> {
    const { data, error } = await this.client
      .from("experiment")
      .select("*")
      .eq("id", id)
      .single();

    handleError(error, `Failed to get experiment with ID ${id}`);
    if (!data) throw new Error(`Experiment with ID ${id} not found.`);
    return mapToExperiment(data);
  }

  async getPublicExperiments(): Promise<Experiment[]> {
    return timeAsync(
      "getPublicExperiments",
      async () => {
        const { data, error } = await this.client
          .from("experiment")
          .select("*");

        handleError(error, "Failed to get public experiments");
        if (!data) throw new Error("unknown error");
        const experiments = data.map((row) => mapToExperiment(row));
        return experiments;
      },
      {},
    );
  }

  async checkExperimentAccess(id: string, userId?: string): Promise<void> {
    if (!userId) {
      throw new Error(`Access denied to experiment with ID ${id}`);
    }

    const { data, error } = await this.client
      .from("experiment")
      .select(
        "workspace_experiments(workspace_id, workspace(user_workspaces(user_id)))",
      )
      .eq("id", id)
      .single();

    handleError(error, `Failed to check access for experiment ID ${id}`);
    if (!data) throw new Error(`Experiment with ID ${id} not found.`);

    const hasWorkspaceAccess = data.workspace_experiments?.some((we) =>
      we.workspace?.user_workspaces?.some((uw) => uw.user_id === userId),
    );

    if (!hasWorkspaceAccess) {
      throw new Error(`Access denied to experiment with ID ${id}`);
    }
  }

  async updateExperiment(
    id: string,
    update: Partial<Experiment>,
  ): Promise<void> {
    const { error } = await this.client
      .from("experiment")
      .update(update as any)
      .eq("id", id);
    handleError(error, `Failed to update experiment with ID ${id}`);
  }

  async deleteExperiment(id: string): Promise<void> {
    const { error } = await this.client
      .from("experiment")
      .delete()
      .eq("id", id);
    handleError(error, `Failed to delete experiment with ID ${id}`);
  }

  async getExperimentsAndMetrics(experimentIds: string[]): Promise<any[]> {
    return timeAsync(
      "db.getExperimentsAndMetrics",
      async () => {
        // First get the experiments
        const { data: experiments, error: experimentsError } = await this.client
          .from("experiment")
          .select(
            "id, name, description, created_at, updated_at, tags, hyperparams",
          )
          .in("id", experimentIds);

        handleError(experimentsError, "Failed to get experiments");

        if (!experiments || experiments.length === 0) return [];

        // Then get all metrics for these experiments
        const { data: metrics, error: metricsError } = await this.client
          .from("metric")
          .select("experiment_id, name, value, step, created_at")
          .in("experiment_id", experimentIds)
          .order("step", { ascending: true })
          .order("created_at", { ascending: true });

        handleError(metricsError, "Failed to get metrics");

        // Group metrics by experiment and metric name
        const metricsByExperiment = new Map();

        if (metrics) {
          metrics.forEach((metric) => {
            if (!metricsByExperiment.has(metric.experiment_id)) {
              metricsByExperiment.set(metric.experiment_id, {});
            }

            const expMetrics = metricsByExperiment.get(metric.experiment_id);
            if (!expMetrics[metric.name]) {
              expMetrics[metric.name] = [];
            }
            expMetrics[metric.name].push(metric.value);
          });
        }

        // Combine experiments with their metrics
        return experiments.map((exp) => ({
          id: exp.id,
          name: exp.name,
          description: exp.description,
          created_at: exp.created_at,
          updated_at: exp.updated_at,
          tags: exp.tags,
          hyperparams: exp.hyperparams,
          metric_dict: metricsByExperiment.get(exp.id) || {},
        }));
      },
      { experimentCount: experimentIds.length.toString() },
    );
  }
}
