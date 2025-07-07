import type { LayoutServerLoad } from "./$types";
import { error } from "@sveltejs/kit";
import { startTimer, generateRequestId } from "$lib/utils/timing";

interface ApiResponse<T> {
  status: number;
  data: T;
}

interface WorkspaceData {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  role: string;
}

interface ExperimentData {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  hyperparams: any[];
}

export const load: LayoutServerLoad = async ({ locals }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("workspaces.load", { requestId });

  try {
    const workspacesResponse =
      await locals.apiClient.get<ApiResponse<WorkspaceData[]>>(
        "/api/workspaces",
      );

    if (workspacesResponse.status !== 200) {
      error(500, "Failed to fetch workspaces");
    }

    const rawWorkspaces = workspacesResponse.data || [];
    const workspaces = rawWorkspaces.map((workspace) => ({
      id: workspace.id,
      name: workspace.name,
      description: workspace.description,
      createdAt: new Date(workspace.created_at),
      role: workspace.role,
    }));

    // TODO: Implement experiments fetching when the endpoint is ready
    const recentExperiments: any[] = [];

    timer.end({
      workspaces_count: workspaces.length,
      user_id: locals.session.user.id,
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

    console.error("Error loading workspaces:", err);

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load workspaces");
  }
};
