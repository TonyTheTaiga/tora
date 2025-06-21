import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";
import { generateRequestId, startTimer } from "$lib/utils/timing";

export const load: PageServerLoad = async ({ params, locals }) => {
  const requestId = generateRequestId();
  const timer = startTimer("experiment.load", { requestId });

  try {
    const userId = locals.user?.id;

    try {
      await locals.dbClient.checkExperimentAccess(params.experimentId, userId);
    } catch (err) {
      timer.end({ error: "Access denied" });
      throw error(403, "Access denied");
    }

    const data = await locals.dbClient.getExperimentsAndMetrics([
      params.experimentId,
    ]);

    if (!data || data.length === 0) {
      timer.end({ error: "Experiment not found" });
      throw error(404, "Experiment not found");
    }

    const item = data[0];
    const experiment = {
      id: item.id,
      name: item.name,
      description: item.description,
      metricData: item.metric_dict,
      tags: item.tags,
      hyperparams: item.hyperparams
        ? (item.hyperparams.map((hp: any) => ({
            key: hp.name || hp.key,
            value: hp.value,
          })) as HyperParam[])
        : [],
      availableMetrics: item.availableMetrics,
      createdAt: new Date(item.created_at),
      updatedAt: new Date(item.updated_at || item.created_at),
      status: item.status || "COMPLETED",
      startedAt: item.started_at || item.created_at,
      endedAt: item.ended_at,
      version: item.version,
    };

    const allMetrics = Object.entries(item.metric_dict || {}).flatMap(
      ([name, values]) =>
        (values as number[]).map((value, index) => ({
          name,
          value,
          step: index,
          experiment_id: item.id,
        })),
    );

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
        scalarMetrics.push(metricList[0]);
      } else {
        timeSeriesNames.push(name);
        timeSeriesMetrics.push(...metricList);
      }
    });

    timer.end({
      experimentId: params.experimentId,
      metricsCount: allMetrics.length,
    });

    return {
      experiment,
      allMetrics,
      scalarMetrics,
      timeSeriesMetrics,
      timeSeriesNames,
    };
  } catch (err) {
    timer.end({ error: err instanceof Error ? err.message : "Unknown error" });
    throw err;
  }
};
