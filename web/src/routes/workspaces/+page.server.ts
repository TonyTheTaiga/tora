import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ fetch, locals }) => {
  const { session } = await locals.safeGetSession();
  const response = await fetch('/api/workspaces')
  const workspaces = await response.json();
  return { workspaces, session };
};
