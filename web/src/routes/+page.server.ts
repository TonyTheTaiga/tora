import type { Actions } from "@sveltejs/kit";
import { fail } from "@sveltejs/kit";
import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { ApiResponse, Workspace } from "$lib/types";

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

      const response = await locals.apiClient.post<ApiResponse<Workspace>>(
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
        createdAt: new Date(workspace.createdAt),
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

  deleteWorkspace: async ({ request, locals, fetch }) => {
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
      const data = await request.formData();
      const workspaceId = data.get("workspaceId") as string;
      const response = await fetch(`/api/workspaces/${workspaceId}`, {
        method: "DELETE",
      });
      console.log(response);
      return { success: true };
    } catch (err) {
      console.error("failed to delete workspace", err);
      fail(500, { error: `failed to delete workspace ${err}` });
    }

    timer.end({});
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
};
