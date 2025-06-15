import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { generateRequestId, startTimer } from "$lib/utils/timing";

function handleError(
  err: unknown,
  timer: ReturnType<typeof startTimer>,
): never {
  const message =
    err instanceof Error ? err.message : "An unknown error occurred.";
  timer.end({ error: message });
  throw error(500, message);
}

export const GET: RequestHandler = async ({ url, locals }) => {
  const requestId = generateRequestId();
  const timer = startTimer("api.experiments.GET", { requestId });

  try {
    const { user, dbClient } = locals;
    if (!user) {
      throw error(401, "Unauthorized");
    }

    const workspaceId = url.searchParams.get("workspace");
    if (!workspaceId) {
      throw error(400, "Missing required 'workspace' parameter");
    }

    const experiments = await dbClient.getExperiments(workspaceId);
    timer.end({
      userId: user.id || "unknown",
      workspaceId,
      experimentCount: experiments.length.toString(),
    });
    return json(experiments);
  } catch (err) {
    if (err instanceof Error && "status" in err) throw err;
    handleError(err, timer);
  }
};

export const POST: RequestHandler = async ({ request, locals, cookies }) => {
  const requestId = generateRequestId();
  const timer = startTimer("api.experiments.POST", { requestId });

  try {
    const { user, dbClient } = locals;
    if (!user) {
      throw error(
        401,
        "Unauthorized: Cannot create an experiment for an anonymous user",
      );
    }

    const data = await request.json();
    const {
      name,
      description,
      tags,
      hyperparams: directHyperparams,
      rawHyperparams,
      visibility = "PRIVATE",
    } = data;

    const workspaceId = data.workspaceId || cookies.get("current_workspace");
    if (!workspaceId) {
      throw error(
        400,
        "A workspace must be specified to create an experiment.",
      );
    }

    let hyperparams = directHyperparams ?? rawHyperparams;
    if (typeof hyperparams === "string") {
      try {
        hyperparams = JSON.parse(hyperparams);
      } catch (e) {
        throw error(400, "Invalid 'hyperparams' format: must be valid JSON.");
      }
    }

    const experiment = await dbClient.createExperiment(user.id, {
      name,
      description,
      hyperparams,
      tags,
      visibility,
      workspaceId,
    });

    timer.end({
      userId: user.id,
      experimentId: experiment.id,
      workspaceId,
    });

    return json({ success: true, experiment }, { status: 201 });
  } catch (err) {
    if (err instanceof Error && "status" in err) throw err;
    handleError(err, timer);
  }
};
