import type { PageServerLoad } from "./$types";
import { apiClient } from "$lib/api";

export const load: PageServerLoad = async () => {
  try {
    const settingsData = await apiClient.get<{
      user: any;
      workspaces: any[];
      apiKeys: any[];
      invitations: any[];
    }>("/api/settings");

    return {
      user: settingsData.user,
      workspaces: settingsData.workspaces,
      apiKeys: settingsData.apiKeys,
      invitations: settingsData.invitations,
    };
  } catch (error) {
    console.error("Error loading settings:", error);
    return {
      user: null,
      workspaces: [],
      apiKeys: [],
      invitations: [],
    };
  }
};
