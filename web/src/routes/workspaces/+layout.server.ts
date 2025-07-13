import type { LayoutServerLoad } from "./$types";
import { error } from "@sveltejs/kit";
import { startTimer, generateRequestId } from "$lib/utils/timing";

interface ApiResponse<T> {
  status: number;
  data: T;
}

interface WorkspaceOverview {
  workspaces: {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    role: string;
    experiment_count: number;
    recent_experiment_count: number;
  }[];
}

export const load: LayoutServerLoad = async ({ locals }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("workspaces.layout.load", { requestId });

  try {
    const workspacesResponse = await locals.apiClient.get<
      ApiResponse<WorkspaceOverview>
    >("/api/dashboard/overview");

    if (workspacesResponse.status !== 200) {
      error(500, "Failed to fetch workspaces");
    }

    const workspacesData = workspacesResponse.data;
    if (!workspacesData) {
      error(500, "No workspaces data received");
    }

    const workspaces = workspacesData.workspaces.map((workspace: any) => ({
      id: workspace.id,
      name: workspace.name,
      description: workspace.description,
      createdAt: new Date(workspace.created_at),
      role: workspace.role,
      experimentCount: workspace.experiment_count,
      recentExperimentCount: workspace.recent_experiment_count,
    }));

    timer.end({
      workspaces_count: workspaces.length,
      user_id: locals.session.user.id,
    });

    return {
      workspaces,
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
