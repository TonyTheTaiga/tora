import { json, type RequestEvent } from "@sveltejs/kit";
import { createMetric, getMetrics, getExperiment } from "$lib/server/database";
import type { Json } from "$lib/server/database.types";

interface MetricInput {
  name: string;
  value: number;
  metadata?: Json;
  step?: number;
}

interface APIResponse {
  success: boolean;
  error?: {
    message: string;
    code: string;
  };
}

export async function POST({
  request,
  params,
  locals,
}: RequestEvent<{ slug: string }, string> & {
  locals: { user: { id: string } | null };
}): Promise<Response> {
  // Check experiment access first
  const userId = locals.user?.id;
  const experimentId = params.slug;

  if (!experimentId?.trim()) {
    return json(
      {
        success: false,
        error: {
          message: "Invalid experiment ID",
          code: "INVALID_ID",
        },
      },
      { status: 400 },
    );
  }

  try {
    // This will throw an error if the user doesn't have access
    await getExperiment(experimentId, userId);
  } catch (error) {
    return json(
      {
        success: false,
        error: {
          message: "Access denied to experiment",
          code: "ACCESS_DENIED",
        },
      },
      { status: 403 },
    );
  }
  try {
    const payload = (await request.json()) as MetricInput;

    if (!payload.name?.trim()) {
      throw new Error("Metric name is required");
    }

    if (payload.value === undefined || payload.value === null) {
      throw new Error("Metric value is required");
    }

    // Experiment ID already validated above

    if (payload.step !== undefined && !Number.isFinite(payload.step)) {
      throw new Error("Step must be a finite number");
    }

    await createMetric({
      experiment_id: experimentId,
      name: payload.name,
      value: payload.value,
      step: payload.step,
      metadata: payload.metadata,
    });

    return json({
      success: true,
    } satisfies APIResponse);
  } catch (error) {
    const statusCode = error instanceof Error ? 400 : 500;

    return json(
      {
        success: false,
        error: {
          message:
            error instanceof Error ? error.message : "Internal server error",
          code: "METRIC_CREATE_FAILED",
        },
      } satisfies APIResponse,
      { status: statusCode },
    );
  }
}

export async function GET({
  params: { slug },
  locals,
}: {
  params: { slug: string };
  locals: { user: { id: string } | null };
}) {
  const userId = locals.user?.id;
  // First verify the user has access to this experiment via getExperiment
  // This will throw an error if they don't have access
  try {
    await getExperiment(slug, userId);
  } catch (error) {
    // Return empty array instead of 403 to avoid UI errors
    return json([], { status: 200 });
  }

  const metrics = await getMetrics(slug);
  return json(metrics);
}
