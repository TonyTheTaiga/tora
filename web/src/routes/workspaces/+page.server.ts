import type { Actions, PageServerLoad } from "./$types";
import type { ApiResponse } from "$lib/types";
import { fail, error } from "@sveltejs/kit";
import { startTimer, generateRequestId } from "$lib/utils/timing";

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

export const load: PageServerLoad = async ({ locals, parent }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("workspaces.page.load", { requestId });

  try {
    const { workspaces } = await parent();

    const [dashboardResponse, workspaceRoles, invitations] = await Promise.all([
      locals.apiClient.get<ApiResponse<DashboardOverview>>(
        "/api/dashboard/overview",
      ),
      locals.apiClient
        .get<Array<{ id: string; name: string }>>("/api/workspace-roles")
        .catch(() => []),
      locals.apiClient.get<any[]>("/api/workspace-invitations").catch(() => []),
    ]);

    if (dashboardResponse.status !== 200) {
      error(500, "Failed to fetch dashboard overview");
    }

    const dashboardData = dashboardResponse.data;
    if (!dashboardData) {
      error(500, "No dashboard data received");
    }

    const recentExperiments = dashboardData.recent_experiments
      .map((exp: any) => ({
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
      workspaceRoles,
      invitations,
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });

    console.error("Error loading workspaces page data:", err);

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load workspaces page data");
  }
};

export const actions: Actions = {
  createWorkspace: async ({ request, locals }) => {
    if (!locals.session) {
      return fail(401, { error: "Authentication required" });
    }

    if (!locals.apiClient) {
      return fail(500, { error: "API client not available" });
    }

    const requestId = generateRequestId();
    const timer = startTimer("workspace.create", {
      requestId,
      userId: locals.session.user.id,
    });

    try {
      const formData = await request.formData();
      const name = formData.get("name") as string;
      const description = formData.get("description") as string;

      if (!name || name.trim().length === 0) {
        timer.end({ error: "Invalid name" });
        return fail(400, { error: "Workspace name is required" });
      }

      const response = await locals.apiClient.post<ApiResponse<WorkspaceData>>(
        "/api/workspaces",
        {
          name: name.trim(),
          description: description?.trim() || null,
        },
      );

      if (response.status !== 201) {
        return fail(500, { error: "Failed to create workspace" });
      }

      const workspace = response.data;

      timer.end({
        workspaceId: workspace.id,
        workspaceName: workspace.name,
      });

      return {
        id: workspace.id,
        name: workspace.name,
        description: workspace.description,
        createdAt: new Date(workspace.created_at),
        role: workspace.role,
      };
    } catch (err) {
      timer.end({
        error: err instanceof Error ? err.message : "Unknown error",
      });

      console.error("Error creating workspace:", err);

      if (err instanceof Error && err.message.includes("401")) {
        return fail(401, { error: "Authentication required" });
      }

      return fail(500, {
        error: "Failed to create workspace. Please try again.",
      });
    }
  },

  deleteWorkspace: async ({ request, locals }) => {
    const data = await request.formData();
    const workspaceId = data.get("workspaceId") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.delete(`/api/workspaces/${workspaceId}`);
      return { success: true };
    } catch (error) {
      console.error("Failed to delete workspace:", error);
      return fail(500, { error: "Failed to delete workspace" });
    }
  },

  sendInvitation: async ({ request, locals }) => {
    const data = await request.formData();
    const workspaceId = data.get("workspaceId") as string;
    const email = data.get("email") as string;
    const roleId = data.get("roleId") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.post("/api/workspace-invitations", {
        workspaceId,
        email,
        roleId,
      });
      return { success: true };
    } catch (error) {
      console.error("Failed to send invitation:", error);
      return fail(500, { error: "Failed to send invitation" });
    }
  },

  respondToInvitation: async ({ request, locals }) => {
    const data = await request.formData();
    const invitationId = data.get("invitationId") as string;
    const action = data.get("action") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.put(
        `/api/workspaces/any/invitations?invitationId=${invitationId}&action=${action}`,
      );
      return { success: true };
    } catch (error) {
      console.error("Failed to respond to invitation:", error);
      return fail(500, { error: "Failed to respond to invitation" });
    }
  },

  leaveWorkspace: async ({ request, locals }) => {
    const data = await request.formData();
    const workspaceId = data.get("workspaceId") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.post(`/api/workspaces/${workspaceId}/leave`);
      return { success: true };
    } catch (error) {
      console.error("Failed to leave workspace:", error);
      return fail(500, { error: "Failed to leave workspace" });
    }
  },
};
