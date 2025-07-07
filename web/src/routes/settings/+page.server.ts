import type { PageServerLoad, Actions } from "./$types";
import { fail } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ locals }) => {
  try {
    const [settingsData, workspaceRoles] = await Promise.all([
      locals.apiClient.get<{
        user: any;
        workspaces: any[];
        apiKeys: any[];
        invitations: any[];
      }>("/api/settings"),
      locals.apiClient
        .get<Array<{ id: string; name: string }>>("/api/workspace-roles")
        .catch(() => []),
    ]);

    return {
      user: settingsData.user,
      workspaces: settingsData.workspaces,
      apiKeys: settingsData.apiKeys,
      invitations: settingsData.invitations,
      workspaceRoles,
      hasElevatedPermissions: locals.apiClient.hasElevatedPermissions(),
    };
  } catch (error) {
    console.error("Error loading settings:", error);
    return {
      user: null,
      workspaces: [],
      apiKeys: [],
      invitations: [],
      workspaceRoles: [],
      hasElevatedPermissions: locals.apiClient.hasElevatedPermissions(),
    };
  }
};

export const actions: Actions = {
  createWorkspace: async ({ request, locals }) => {
    const data = await request.formData();
    const name = data.get("name") as string;
    const description = data.get("description") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.post("/api/workspaces", {
        name,
        description,
      });
      return { success: true };
    } catch (error) {
      console.error("Failed to create workspace:", error);
      return fail(500, { error: "Failed to create workspace" });
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

  createApiKey: async ({ request, locals }) => {
    const data = await request.formData();
    const name = data.get("name") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      const result = await locals.apiClient.post<{ data: { key: string } }>(
        "/api/api-keys",
        {
          name,
        },
      );
      return { success: true, key: result.data.key };
    } catch (error) {
      console.error("Failed to create API key:", error);
      return fail(500, { error: "Failed to create API key" });
    }
  },

  revokeApiKey: async ({ request, locals }) => {
    const data = await request.formData();
    const keyId = data.get("keyId") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.delete(`/api/api-keys/${keyId}`);
      return { success: true };
    } catch (error) {
      console.error("Failed to revoke API key:", error);
      return fail(500, { error: "Failed to revoke API key" });
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
};
