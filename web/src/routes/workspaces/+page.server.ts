import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ fetch, locals, parent }) => {
  const { session } = await locals.safeGetSession();
  const { currentWorkspace } = await parent();
  const response = await fetch("/api/workspaces");
  const workspaces = await response.json();
  return { workspaces, session, currentWorkspace };
};
