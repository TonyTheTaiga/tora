import { error } from "@sveltejs/kit";
import type { PageLoad } from "./$types";

export const load: PageLoad = async ({ params, fetch, parent }) => {
  const parentData = await parent();
  const workspace = parentData.workspaces?.find((w) => w.id === params.slug);

  try {
    // Load experiment details
    const experimentResponse = await fetch(
      `/api/experiments/${params.experimentId}`,
    );
    if (!experimentResponse.ok) {
      throw error(404, "Experiment not found");
    }
    const experiment = await experimentResponse.json();

    // Load metrics for the experiment
    let allMetrics = [];
    try {
      const metricsResponse = await fetch(
        `/api/experiments/${params.experimentId}/metrics`,
      );
      if (metricsResponse.ok) {
        allMetrics = await metricsResponse.json();
      }
    } catch (err) {
      console.warn("Failed to load metrics:", err);
    }

    // Group metrics by name and classify as scalar vs time series
    const metricsByName = new Map();
    allMetrics.forEach((metric: any) => {
      if (!metricsByName.has(metric.name)) {
        metricsByName.set(metric.name, []);
      }
      metricsByName.get(metric.name).push(metric);
    });

    const scalarMetrics: any[] = [];
    const timeSeriesMetrics: any[] = [];
    const timeSeriesNames: string[] = [];

    metricsByName.forEach((metricList: any[], name: string) => {
      if (metricList.length === 1) {
        // Single data point = scalar metric
        scalarMetrics.push(metricList[0]);
      } else {
        // Multiple data points = time series metric
        timeSeriesNames.push(name);
        timeSeriesMetrics.push(...metricList);
      }
    });

    // Load experiment files/artifacts if available
    let files = [];
    try {
      const filesResponse = await fetch(
        `/api/experiments/${params.experimentId}/files`,
      );
      if (filesResponse.ok) {
        files = await filesResponse.json();
      }
    } catch (err) {
      console.warn("Failed to load files:", err);
    }

    return {
      experiment,
      allMetrics,
      scalarMetrics,
      timeSeriesMetrics,
      timeSeriesNames,
      files,
      workspace,
    };
  } catch (err) {
    throw error(500, "Failed to load experiment data");
  }
};
