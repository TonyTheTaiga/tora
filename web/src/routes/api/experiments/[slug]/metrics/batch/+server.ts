import { json, type RequestEvent } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { broadcastMetric } from "$lib/server/broadcast";
import type { Json } from "$lib/server/database.types";
import type { Metric } from "$lib/types";

interface MetricInput {
  name: string;
  value: number;
  step?: number;
  metadata?: Json;
}

interface APIError {
  message: string;
  code: string;
  status: number;
}

export const POST: RequestHandler = async ({ request, params, locals }) => {
  const userId = locals.user?.id;
  const experimentId = params.slug;

  if (!experimentId?.trim()) {
    throw new Error("Invalid experiment ID");
  }

  try {
    await locals.dbClient.checkExperimentAccess(experimentId, userId);
  } catch (error) {
    return json(
      {
        message: "Access denied to experiment",
        code: "ACCESS_DENIED",
      },
      { status: 403 },
    );
  }
  try {
    const metrics = (await request.json()) as MetricInput[];
    if (!Array.isArray(metrics)) {
      throw new Error("Invalid input: expected array of metrics");
    }

    if (!metrics.every(isValidMetric)) {
      throw new Error("Invalid metric format");
    }

    const experimentId = params.slug;
    if (!experimentId?.trim()) {
      throw new Error("Invalid experiment ID");
    }

    const currentTime = new Date().toISOString();
    const finalMetrics = metrics.map((data) => ({
      ...data,
      experiment_id: experimentId,
      created_at: currentTime,
    })) as Metric[];

    await locals.dbClient.batchCreateMetric(finalMetrics);

    for (const m of metrics) {
      broadcastMetric(experimentId, JSON.stringify(m));
    }

    return json({
      success: true,
      count: metrics.length,
      experimentId,
    });
  } catch (error) {
    const apiError: APIError = {
      message:
        error instanceof Error ? error.message : "Unknown error occurred",
      code: "METRIC_CREATE_ERROR",
      status: 400,
    };

    return json(apiError, { status: apiError.status });
  }
};

function isValidMetric(metric: unknown): metric is MetricInput {
  if (!metric || typeof metric !== "object") return false;

  const m = metric as MetricInput;

  return (
    typeof m.name === "string" &&
    m.name.trim().length > 0 &&
    typeof m.value === "number" &&
    (m.step === undefined || typeof m.step === "number") &&
    (m.metadata === undefined || typeof m.metadata === "object")
  );
}
