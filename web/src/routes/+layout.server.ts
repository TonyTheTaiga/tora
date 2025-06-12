import type { Workspace } from "$lib/types";
import type { LayoutServerLoad } from "./$types";

export const load: LayoutServerLoad = async ({ locals, cookies }) => {
  const { session, user } = await locals.safeGetSession();

  let currentWorkspace: Workspace | null = null;
  let userWorkspaces: Workspace[] = [];

  if (user) {
    userWorkspaces = await locals.dbClient.getWorkspacesV2(user.id, [
      "OWNER", 
      "ADMIN", 
      "EDITOR", 
      "VIEWER"
    ]);
    const workspaceId = cookies.get("current_workspace");

    if (workspaceId) {
      currentWorkspace =
        userWorkspaces.find((w) => w.id === workspaceId) || null;
    }

    if (!currentWorkspace) {
      currentWorkspace = await locals.dbClient.getOrCreateDefaultWorkspace(
        user.id,
      );
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
