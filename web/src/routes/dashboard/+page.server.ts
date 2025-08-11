import type { Actions } from "@sveltejs/kit";
import type { ApiResponse, Workspace } from "$lib/types";
import { fail, error } from "@sveltejs/kit";
import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ locals, fetch }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  try {
    const requestId = generateRequestId();
    const timer = startTimer("dashboard.page.load", { requestId });
    const response = await fetch("/api/workspaces");
    const apiResponse: ApiResponse<Workspace[]> = await response.json();
    const workspaces = apiResponse.data.map((workspace: any) => ({
      id: workspace.id,
      name: workspace.name,
      description: workspace.description,
      createdAt: new Date(workspace.createdAt),
      role: workspace.role,
    }));

    timer.end({
      workspaces_count: workspaces.length,
      user_id: locals.session.user.id,
    });

    return { workspaces };
  } catch (err) {
    if (err instanceof Error && "status" in err) {
      throw err;
    }
    throw error(500, "Failed to load workspaces");
  }
};
