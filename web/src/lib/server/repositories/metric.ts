import type { Metric } from "$lib/types";
import { timeAsync } from "$lib/utils/timing";
import { BaseRepository, handleError } from "./base";

export class MetricRepository extends BaseRepository {
  async getMetrics(experimentId: string): Promise<Metric[]> {
    return timeAsync(
      "db.getMetrics",
      async () => {
        const { data, error } = await this.client
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
  }

  async createMetric(metric: Metric): Promise<void> {
    const { error } = await this.client.from("metric").insert(metric);
    handleError(error, "Failed to write metric");
  }

  async batchCreateMetric(metrics: Metric[]): Promise<void> {
    const { error } = await this.client.from("metric").insert(metrics);
    // For production apps, consider a more robust retry mechanism (e.g., using p-retry)
    handleError(error, "Failed to batch write metrics");
  }
}