import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { generateRequestId, startTimer } from "$lib/utils/timing";

export const GET: RequestHandler = async ({ params, locals }) => {
  const requestId = generateRequestId();
  const timer = startTimer("api.experimentRef.GET", { requestId, experimentId: params.slug });
  
  try {
    const references = await locals.dbClient.getReferenceChain(params.slug);
    timer.end({ referenceCount: references.length });
    return json(references.map((experiment) => experiment.id));
  } catch (error) {
    timer.end({ error: error instanceof Error ? error.message : "Unknown error" });
    throw error;
  }
};

export const POST: RequestHandler = async ({ params, request, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ message: "Authentication required" }, { status: 401 });
  }

  try {
    const { referenceId } = await request.json();
    try {
      await locals.dbClient.checkExperimentAccess(params.slug, userId);
      await locals.dbClient.checkExperimentAccess(referenceId, userId);
    } catch (accessError) {
      return json(
        { message: "Access denied to one or both experiments" },
        { status: 403 },
      );
    }

    await locals.dbClient.createReference(params.slug, referenceId);
    return json({ message: "Reference created successfully" }, { status: 201 });
  } catch (error) {
    return json(
      {
        message:
          error instanceof Error
            ? error.message
            : "Unknown error creating reference",
      },
      { status: 500 },
    );
  }
};
