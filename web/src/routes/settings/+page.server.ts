import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { PageServerLoad, Actions } from "./$types";
import { error } from "@sveltejs/kit";
import type { Workspace, ApiKey } from "$lib/types";

export const load: PageServerLoad = async ({ fetch, locals, parent, url }) => {
  const { session, user } = await locals.safeGetSession();
  const { currentWorkspace } = await parent();

  const requestId = generateRequestId();
  const timer = startTimer("page.home.load", { requestId });
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

  return { workspaces, apiKeys, session, currentWorkspace };
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
    console.log(id);

    const response = await fetch(`/api/workspaces/${id}`, {
      method: "DELETE",
    });

    return await response.json();
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
};
