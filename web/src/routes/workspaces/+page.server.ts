import type { Actions, PageServerLoad } from "./$types";
import { error } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ fetch, locals, parent }) => {
  const { session } = await locals.safeGetSession();
  const { currentWorkspace } = await parent();
  const response = await fetch("/api/workspaces");
  const workspaces = await response.json();
  return { workspaces, session, currentWorkspace };
};

export const actions: Actions = {
  create: async ({ request, fetch }) => {
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
