import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";
import { generateRequestId, startTimer } from "$lib/utils/timing";

interface ApiResponse<T> {
  status: number;
  data: T;
}

interface ExperimentData {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  hyperparams: any[];
  workspace_id: string;
  available_metrics: string[];
}

interface MetricData {
  id: number;
  experiment_id: string;
  name: string;
  value: number;
  step?: number;
  metadata?: any;
  created_at: string;
}

export interface ExperimentWithMetrics {
  id: string;
  name: string;
  description: string;
  metricData: Record<string, number[]>;
  tags: string[];
  hyperparams: HyperParam[] | null;
  createdAt: Date;
  workspaceId: string;
}

export const load: PageServerLoad = async ({ url, locals }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("page.compare.load", { requestId });

  try {
    const idsParam = url.searchParams.get("ids");
    if (!idsParam) {
      error(400, "Missing 'ids' query parameter");
    }

    const ids = idsParam.split(",").filter(Boolean);
    if (ids.length === 0) {
      error(400, "No valid experiment IDs provided");
    }

    // Fetch all experiments and their metrics
    const experimentPromises = ids.map(async (id) => {
      try {
        // Fetch experiment details
        const experimentResponse = await locals.apiClient.get<
          ApiResponse<ExperimentData>
        >(`/api/experiments/${id}`);

        if (experimentResponse.status !== 200) {
          console.warn(
            `Failed to fetch experiment ${id}: status ${experimentResponse.status}`,
          );
          return null;
        }

        const experimentData = experimentResponse.data;
        if (!experimentData) {
          console.warn(`No data for experiment ${id}`);
          return null;
        }

        // Fetch metrics for this experiment
        let metricsData: MetricData[] = [];
        try {
          const metricsResponse = await locals.apiClient.get<
            ApiResponse<MetricData[]>
          >(`/api/experiments/${id}/metrics`);

          if (metricsResponse.status === 200) {
            metricsData = metricsResponse.data || [];
          }
        } catch (metricsErr) {
          console.warn(
            `Failed to fetch metrics for experiment ${id}:`,
            metricsErr,
          );
          // Continue without metrics
        }

        // Process metrics into the expected format
        const metricData: Record<string, number[]> = {};
        const metricsByName = new Map<string, MetricData[]>();

        metricsData.forEach((metric) => {
          if (!metricsByName.has(metric.name)) {
            metricsByName.set(metric.name, []);
          }
          metricsByName.get(metric.name)!.push(metric);
        });

        metricsByName.forEach((metricList, name) => {
          metricData[name] = metricList
            .sort((a, b) => (a.step || 0) - (b.step || 0))
            .map((m) => m.value);
        });

        return {
          id: experimentData.id,
          name: experimentData.name,
          description: experimentData.description,
          metricData,
          tags: experimentData.tags || [],
          hyperparams: experimentData.hyperparams
            ? (experimentData.hyperparams.map((hp: any) => ({
                key: hp.name || hp.key,
                value: hp.value,
              })) as HyperParam[])
            : null,
          createdAt: new Date(experimentData.created_at),
          workspaceId: experimentData.workspace_id,
        } as ExperimentWithMetrics;
      } catch (err) {
        console.warn(`Error fetching experiment ${id}:`, err);
        return null;
      }
    });

    const experimentResults = await Promise.all(experimentPromises);
    const experiments = experimentResults
      .filter((exp): exp is ExperimentWithMetrics => exp !== null)
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

    if (experiments.length === 0) {
      error(404, "No valid experiments found for comparison");
    }

    timer.end({
      experimentCount: experiments.length,
      requestedIds: ids.length,
      successfulFetches: experiments.length,
    });

    return { experiments };
  } catch (err) {
    timer.end({ error: err instanceof Error ? err.message : "Unknown error" });

    if (err instanceof Error) {
      if (err.message.includes("401")) {
        error(401, "Authentication required");
      }
      if (err.message.includes("400")) {
        throw err; // Re-throw 400 errors as they are
      }
    }

    console.error("Error loading comparison data:", err);
    error(500, "Failed to load comparison data");
  }
};
