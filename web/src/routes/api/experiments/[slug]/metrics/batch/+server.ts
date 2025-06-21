import { json, error, type RequestHandler } from "@sveltejs/kit";
import type { Json } from "$lib/server/database.types";
import type { Metric } from "$lib/types";

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

    if (!Array.isArray(payload)) {
      throw new Error("Request body must be an array of metrics.");
    }

    const finalMetrics = payload.map((metric: unknown, index: number) => {
      if (!metric || typeof metric !== "object") {
        throw new Error(
          `Invalid metric at index ${index}: Expected an object.`,
        );
      }

      const { name, value, step } = metric as MetricInput;

      if (typeof name !== "string" || !name.trim()) {
        throw new Error(
          `Invalid metric at index ${index}: 'name' must be a non-empty string.`,
        );
      }
      if (!Number.isFinite(value)) {
        throw new Error(
          `Invalid metric at index ${index}: 'value' must be a finite number.`,
        );
      }
      if (step !== undefined && !Number.isFinite(step)) {
        throw new Error(
          `Invalid metric at index ${index}: 'step' must be a finite number if provided.`,
        );
      }

      return {
        ...(metric as MetricInput),
        experiment_id: experimentId,
      };
    });

    if (finalMetrics.length > 0) {
      await dbClient.batchCreateMetric(finalMetrics as Metric[]);
    }

    return json(
      {
        success: true,
        count: finalMetrics.length,
        experimentId,
      },
      { status: 201 },
    );
  } catch (err) {
    const isClientError =
      err instanceof SyntaxError ||
      (err instanceof Error && !(err instanceof TypeError));
    const statusCode = isClientError ? 400 : 500;
    const message = isClientError
      ? err.message
      : "An unexpected internal error occurred.";

    throw error(statusCode, { message });
  }
};
