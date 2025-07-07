import type { LayoutServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import { error } from "@sveltejs/kit";

export const load: LayoutServerLoad = async ({ locals }) => {
  if (!locals.user) {
    error(501, "user required");
  }
  const requestId = generateRequestId();
  const timer = startTimer("workspaces.load", {
    requestId,
  });

  try {
    const { workspaces, experiments: allExperiments } =
      await locals.dbClient.getWorkspacesAndExperiments(locals.user.id, [
        "OWNER",
        "ADMIN",
        "EDITOR",
        "VIEWER",
      ]);

    allExperiments.sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );

    const recentExperiments = allExperiments.slice(0, 10);

    timer.end({
      user_id: locals.user.id,
      workspaces_count: workspaces.length,
      experiments_count: allExperiments.length,
    });

    return {
      workspaces,
      recentExperiments,
      recentWorkspaces: workspaces.slice(0, 5),
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
