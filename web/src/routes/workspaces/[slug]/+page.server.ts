import type { PageServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { Experiment } from "$lib/types";


export const load: PageServerLoad = async ({ locals, params, parent }) => {
  const requestId = generateRequestId();
  const { workspaces } = await parent();

  const timer = startTimer("workspaces.load", {
    requestId,
  });
  const workspaceId = params.slug;
  try {
    const experiments: Experiment[] =
      await locals.dbClient.getExperiments(workspaceId);

    const currentWorkspace = workspaces.find((w) => w.id === workspaceId);
    timer.end({ workspace_id: workspaceId });
    return { experiments, currentWorkspace };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};

