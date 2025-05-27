import type { Workspace } from "$lib/types";
import type { LayoutServerLoad } from "./$types";
import {
  getWorkspaces,
  getOrCreateDefaultWorkspace,
} from "$lib/server/database";

export const load: LayoutServerLoad = async ({ locals, cookies }) => {
  const { session, user } = await locals.safeGetSession();

  let currentWorkspace: Workspace | null = null;
  let userWorkspaces: Workspace[] = [];

  if (user) {
    userWorkspaces = await getWorkspaces(user.id);
    const workspaceId = cookies.get("current_workspace");

    if (workspaceId) {
      currentWorkspace =
        userWorkspaces.find((w) => w.id === workspaceId) || null;
    }

    if (!currentWorkspace) {
      currentWorkspace = await getOrCreateDefaultWorkspace(user.id);
      userWorkspaces.push(currentWorkspace);

      cookies.set("current_workspace", currentWorkspace.id, {
        path: "/",
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        maxAge: 60 * 60 * 24 * 30,
      });
    }
  }
  return {
    session,
    cookies: cookies.getAll(),
    currentWorkspace,
    userWorkspaces,
  };
};
