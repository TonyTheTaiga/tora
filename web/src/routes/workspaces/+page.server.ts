import type { Actions } from "./$types";
import { error, fail } from "@sveltejs/kit";
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
      const name = formData.get("workspace-name") as string;
      const description = formData.get("workspace-description") as string;

      if (!name || name.trim().length === 0) {
        timer.end({ error: "Invalid name" });
        return fail(400, { error: "Workspace name is required" });
      }

      const response = await locals.apiClient.post<ApiResponse<WorkspaceData>>(
        "/api/workspaces",
        {
          "workspace-name": name.trim(),
          "workspace-description": description?.trim() || null,
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
};
