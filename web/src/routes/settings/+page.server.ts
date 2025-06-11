import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { PageServerLoad, Actions } from "./$types";
import { error } from "@sveltejs/kit";
import type { Workspace } from "$lib/types";

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

  // timer.end({});
  return { workspaces, session, currentWorkspace };
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
};
