import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { PageServerLoad } from "./$types";
import type { Experiment, Workspace } from "$lib/types";

export const load: PageServerLoad = async ({ fetch, locals, parent, url }) => {
  const requestId = generateRequestId();
  const timer = startTimer("home.load", {
    requestId,
  });
  
  try {
    const { session, user } = await locals.safeGetSession();
    
    if (!user) {
      timer.end({});
      return {};
    }

    const { workspaces, experiments: allExperiments } = await locals.dbClient.getWorkspacesAndExperiments(user.id, [
      "OWNER",
      "ADMIN", 
      "EDITOR",
      "VIEWER",
    ]);

    allExperiments.sort(
      (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );

    const recentExperiments = allExperiments.slice(0, 10);

    const stats = {
      totalExperiments: allExperiments.length,
      totalWorkspaces: workspaces.length,
      recentExperimentsCount: allExperiments.filter(exp => {
        const oneWeekAgo = new Date();
        oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
        return new Date(exp.createdAt) > oneWeekAgo;
      }).length
    };

    timer.end({ 
      user_id: user.id,
      workspaces_count: workspaces.length,
      experiments_count: allExperiments.length
    });
    
    return {
      workspaces: workspaces.slice(0, 5),
      recentExperiments,
      stats
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
