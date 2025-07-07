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
  workspace_id: string;
  available_metrics: string[];
}

interface DashboardOverview {
  workspaces: {
    id: string;
    name: string;
    description: string | null;
    created_at: string;
    role: string;
    experiment_count: number;
    recent_experiment_count: number;
  }[];
  recent_experiments: ExperimentData[];
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
    // Use the new optimized dashboard endpoint
    const dashboardResponse =
      await locals.apiClient.get<ApiResponse<DashboardOverview>>(
        "/api/dashboard/overview",
      );

    if (dashboardResponse.status !== 200) {
      error(500, "Failed to fetch dashboard overview");
    }

    const dashboardData = dashboardResponse.data;
    if (!dashboardData) {
      error(500, "No dashboard data received");
    }

    const workspaces = dashboardData.workspaces.map((workspace) => ({
      id: workspace.id,
      name: workspace.name,
      description: workspace.description,
      createdAt: new Date(workspace.created_at),
      role: workspace.role,
      experimentCount: workspace.experiment_count,
      recentExperimentCount: workspace.recent_experiment_count,
    }));

    const recentExperiments = dashboardData.recent_experiments
      .map((exp) => ({
        id: exp.id,
        name: exp.name,
        description: exp.description || "",
        hyperparams: exp.hyperparams || [],
        tags: exp.tags || [],
        createdAt: new Date(exp.created_at),
        updatedAt: new Date(exp.updated_at),
        availableMetrics: exp.available_metrics || [],
        workspaceId: exp.workspace_id,
      }))
      .slice(0, 5);

    timer.end({
      workspaces_count: workspaces.length,
      recent_experiments_count: recentExperiments.length,
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

    console.error("Error loading dashboard overview:", err);

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load dashboard overview");
  }
};
