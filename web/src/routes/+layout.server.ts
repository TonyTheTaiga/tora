import type { Workspace } from "$lib/types";
import type { LayoutServerLoad } from "./$types";
import {
  getWorkspaces,
  getOrCreateDefaultWorkspace,
} from "$lib/server/database";

export const load: LayoutServerLoad = async ({ locals, cookies }) => {
  const { session, user } = await locals.safeGetSession();
  console.log("🏠 Layout load - User ID:", user?.id);

  let currentWorkspace: Workspace | null = null;
  let userWorkspaces: Workspace[] = [];

  if (user) {
    userWorkspaces = await getWorkspaces(user.id);
    console.log("📚 User workspaces:", userWorkspaces.map(w => ({ id: w.id, name: w.name })));
    
    const workspaceId = cookies.get("current_workspace");
    console.log("🍪 Cookie workspace ID:", workspaceId);

    if (workspaceId) {
      currentWorkspace = userWorkspaces.find((w) => w.id === workspaceId) || null;
      console.log("🎯 Found workspace:", currentWorkspace?.name || "NOT FOUND");
    }

    if (!currentWorkspace) {
      console.log("🆕 Creating/getting default workspace");
      currentWorkspace = await getOrCreateDefaultWorkspace(user.id);
      cookies.set("current_workspace", currentWorkspace.id, {
        path: "/",
        httpOnly: true,
        secure: process.env.NODE_ENV === "production",
        sameSite: "lax",
        maxAge: 60 * 60 * 24 * 30,
      });
      console.log("🆕 Default workspace set:", currentWorkspace.name);

      userWorkspaces = await getWorkspaces(user.id);
    }
  }

  console.log("✅ Final current workspace:", currentWorkspace?.name);
  return {
    session,
    cookies: cookies.getAll(),
    currentWorkspace,
    userWorkspaces,
  };
};
