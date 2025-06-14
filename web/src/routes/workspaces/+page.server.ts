import type { Actions, PageServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import { error, redirect, fail } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ locals }) => {
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

export const actions: Actions = {
  createWorkspace: async ({ request, locals }) => {
    if (!locals.user) {
      return fail(401, { error: "Authentication required" });
    }

    const requestId = generateRequestId();
    const timer = startTimer("workspace.create", {
      requestId,
      userId: locals.user.id,
    });

    try {
      const formData = await request.formData();
      const name = formData.get("workspace-name") as string;
      const description = formData.get("workspace-description") as string;

      if (!name || name.trim().length === 0) {
        timer.end({ error: "Invalid name" });
        return fail(400, { error: "Workspace name is required" });
      }

      const workspace = await locals.dbClient.createWorkspace(
        name.trim(),
        description?.trim() || null,
        locals.user.id,
      );

      timer.end({
        workspaceId: workspace.id,
        workspaceName: workspace.name,
      });

      throw redirect(302, `/workspaces/${workspace.id}`);
    } catch (err) {
      if (err instanceof Response) {
        throw err;
      }

      timer.end({
        error: err instanceof Error ? err.message : "Unknown error",
      });

      return fail(500, {
        error: "Failed to create workspace. Please try again.",
      });
    }
  },
};
