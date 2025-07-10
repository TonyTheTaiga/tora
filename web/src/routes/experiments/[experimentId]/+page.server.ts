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

export const load: PageServerLoad = async ({ params, locals }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("experiment.load", { requestId });

  try {
    // Fetch experiment details
    const experimentResponse = await locals.apiClient.get<
      ApiResponse<ExperimentData>
    >(`/api/experiments/${params.experimentId}`);

    if (experimentResponse.status === 404) {
      timer.end({ error: "Experiment not found" });
      error(404, "Experiment not found");
    }

    if (experimentResponse.status !== 200) {
      timer.end({ error: "Failed to fetch experiment" });
      error(500, "Failed to fetch experiment");
    }

    const experimentData = experimentResponse.data;
    if (!experimentData) {
      timer.end({ error: "No experiment data received" });
      error(404, "Experiment not found");
    }

    // Fetch metrics for the experiment
    let metricsData: MetricData[] = [];
    try {
      const metricsResponse = await locals.apiClient.get<
        ApiResponse<MetricData[]>
      >(`/api/experiments/${params.experimentId}/metrics`);

      if (metricsResponse.status === 200) {
        metricsData = metricsResponse.data || [];
      }
    } catch (metricsErr) {
      console.error("Error fetching metrics:", metricsErr);
      // Continue without metrics if they fail to load
    }

    const experiment = {
      id: experimentData.id,
      name: experimentData.name,
      description: experimentData.description,
      tags: experimentData.tags || [],
      hyperparams: experimentData.hyperparams
        ? (experimentData.hyperparams.map((hp: any) => ({
            key: hp.key,
            value: hp.value,
          })) as HyperParam[])
        : [],
      availableMetrics: experimentData.available_metrics || [],
      createdAt: new Date(experimentData.created_at),
      updatedAt: new Date(experimentData.updated_at),
      workspaceId: experimentData.workspace_id,
    };

    // Process metrics data
    const metricsByName = new Map<string, MetricData[]>();
    metricsData.forEach((metric) => {
      if (!metricsByName.has(metric.name)) {
        metricsByName.set(metric.name, []);
      }
      metricsByName.get(metric.name)!.push(metric);
    });

    const scalarMetrics: MetricData[] = [];
    const timeSeriesMetrics: MetricData[] = [];
    const timeSeriesNames: string[] = [];

    metricsByName.forEach((metricList, name) => {
      if (metricList.length === 1) {
        scalarMetrics.push(metricList[0]);
      } else {
        timeSeriesNames.push(name);
        timeSeriesMetrics.push(
          ...metricList.sort((a, b) => (a.step || 0) - (b.step || 0)),
        );
      }
    });

    // Create metric data structure for charts
    const metricData: { [key: string]: number[] } = {};
    metricsByName.forEach((metricList, name) => {
      metricData[name] = metricList
        .sort((a, b) => (a.step || 0) - (b.step || 0))
        .map((m) => m.value);
    });

    timer.end({
      experimentId: params.experimentId,
      metricsCount: metricsData.length,
      scalarMetricsCount: scalarMetrics.length,
      timeSeriesCount: timeSeriesNames.length,
    });

    return {
      experiment: {
        ...experiment,
        metricData,
      },
      allMetrics: metricsData,
      scalarMetrics,
      timeSeriesMetrics,
      timeSeriesNames,
    };
  } catch (err) {
    timer.end({ error: err instanceof Error ? err.message : "Unknown error" });
    console.error("Error loading experiment:", err);

    if (err instanceof Error && err.message.includes("404")) {
      error(404, "Experiment not found");
    }

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load experiment");
  }
};
