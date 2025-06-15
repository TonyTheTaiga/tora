import type { Actions } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import { redirect, fail } from "@sveltejs/kit";

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
