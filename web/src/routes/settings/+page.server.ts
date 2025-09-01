import type { PageServerLoad, Actions } from "./$types";
import { fail } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ locals }) => {
  try {
    const settingsData = await locals.apiClient.get<{
      user: any;
      apiKeys: any[];
    }>("/settings");

    return {
      user: settingsData.user,
      apiKeys: settingsData.apiKeys,
      hasElevatedPermissions: locals.apiClient.hasElevatedPermissions(),
    };
  } catch (error) {
    console.error("Error loading settings:", error);
    return {
      user: null,
      apiKeys: [],
      hasElevatedPermissions: locals.apiClient.hasElevatedPermissions(),
    };
  }
};

export const actions: Actions = {
  createApiKey: async ({ request, locals }) => {
    const data = await request.formData();
    const name = data.get("name") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      const result = await locals.apiClient.post<{ data: { key: string } }>(
        "/api-keys",
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
      await locals.apiClient.delete(`/api-keys/${keyId}`);
      return { success: true };
    } catch (error) {
      console.error("Failed to revoke API key:", error);
      return fail(500, { error: "Failed to revoke API key" });
    }
  },
};
