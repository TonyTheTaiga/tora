import { json, error, type RequestHandler } from "@sveltejs/kit";
import type { Json } from "$lib/server/database.types";
import { generateRequestId, startTimer } from "$lib/utils/timing";

interface MetricInput {
  name: string;
  value: number;
  step?: number;
  metadata?: Json;
}

export const POST: RequestHandler = async ({ request, params, locals }) => {
  const { slug: experimentId } = params;
  const { dbClient } = locals;

  if (!experimentId) {
    throw error(400, {
      message: "Experiment ID is required",
    });
  }

  try {
    const payload = await request.json();

    const { name, value, step } = payload;
    if (!name?.trim()) throw new Error("Metric name is required");
    if (value == null) throw new Error("Metric value is required");
    if (step !== undefined && !Number.isFinite(step)) {
      throw new Error("Step must be a finite number");
    }

    await dbClient.createMetric({
      experiment_id: experimentId,
      ...payload,
    });

    return json({ success: true }, { status: 201 });
  } catch (err) {
    const isValidationError =
      err instanceof Error && !(err instanceof SyntaxError);
    const statusCode = isValidationError ? 400 : 500;
    const message = isValidationError ? err.message : "Internal server error";
    throw error(statusCode, { message });
  }
};

export const GET: RequestHandler = async ({ params, locals }) => {
  const { slug: experimentId } = params;
  const { dbClient } = locals;

  if (!experimentId) {
    return error(401, { message: "Experiment ID is required" });
  }

  const requestId = generateRequestId();
  const timer = startTimer("api.metrics.GET", { requestId, experimentId });

  try {
    const metrics = await dbClient.getMetrics(experimentId);
    timer.end({ metricsCount: metrics.length });

    return json(metrics);
  } catch (err) {
    timer.end({ error: err instanceof Error ? err.message : "Unknown error" });
    throw err;
  }
};
