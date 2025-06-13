import type { PageServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";

export const load: PageServerLoad = async ({ locals, params, parent }) => {
  const requestId = generateRequestId();
  const { workspaces } = await parent();

  const timer = startTimer("workspaces.load", {
    requestId,
  });
  const workspaceId = params.slug;
  try {
    const experiments = await locals.dbClient.getExperiments(workspaceId);
    const workspace = workspaces.find((w) => w.id === workspaceId);

    timer.end({ workspace_id: workspaceId });
    return { experiments, workspace };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
