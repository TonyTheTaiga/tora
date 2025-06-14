import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { PageServerLoad, Actions } from "./$types";
import { error, redirect } from "@sveltejs/kit";
import type { Workspace, ApiKey } from "$lib/types";

export const load: PageServerLoad = async ({ fetch, locals, parent, url }) => {
  const { session, user } = await locals.safeGetSession();

  const requestId = generateRequestId();
  const timer = startTimer("settings.load", { requestId });
  if (!user) {
    const experiments = new Array();
    return { experiments, session };
  }

  const response = await fetch("/api/workspaces");
  const workspaces: Workspace[] = await response.json();

  let apiKeys: ApiKey[] = [];
  try {
    const response = await fetch("/api/keys");
    if (response.ok) {
      const data = await response.json();
      apiKeys = data.keys;
    }
  } catch (err) {
    console.error("Error fetching API keys:", err);
  }

  timer.end({});
  return { workspaces, apiKeys, session };
};

export const actions: Actions = {
  createWorkspace: async ({ request, fetch }) => {
    const fd = await request.formData();
    const rawName = fd.get("name");
    const rawDescription = fd.get("description");

    if (typeof rawName !== "string" || !rawName.trim()) {
      throw error(400, { message: "name required" });
    }
    const description =
      typeof rawDescription === "string" && rawDescription.trim()
        ? rawDescription
        : null;

    const res = await fetch("/api/workspaces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: rawName.trim(), description }),
    });

    return await res.json();
  },

  deleteWorkspace: async ({ request, fetch }) => {
    const formData = await request.formData();
    const { id } = Object.fromEntries(formData.entries());
    if (!id) {
      throw error(400, "ID required to delete workspace");
    }
    await fetch(`/api/workspaces/${id}`, {
      method: "DELETE",
    });

    return redirect(300, "/settings");
  },

  removeSharedWorkspace: async ({ request, fetch }) => {
    const formData = await request.formData();
    const { userId, workspaceId } = Object.fromEntries(formData.entries());

    if (!(userId && workspaceId)) {
      throw error(
        400,
        "userId and workspaceId required to remove shared workspace",
      );
    }

    const response = await fetch(
      `/api/workspaces/${workspaceId}/members/${userId}`,
      {
        method: "DELETE",
      },
    );

    if (!response.ok) {
      throw error(500, "Failed to leave workspace");
    }

    return redirect(300, "/settings");
  },

  createApiKey: async ({ request, fetch }) => {
    const formData = await request.formData();
    const { name } = Object.fromEntries(formData.entries());
    if (!name) {
      throw error(400, "A unique name is required for your api key");
    }

    const response = await fetch("/api/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name }),
    });

    return await response.json();
  },

  revokeApiKey: async ({ request, fetch }) => {
    const formData = await request.formData();
    const { id } = Object.fromEntries(formData.entries());
    if (!id) {
      throw error(400, "ID reuired to revoke api key");
    }

    await fetch(`/api/keys/${id}`, { method: "DELETE" });
    return redirect(300, "/settings");
  },
};
