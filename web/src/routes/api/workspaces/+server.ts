import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ locals }) => {
  if (!locals.user) {
    return json([]);
  }
  const workspaces = await locals.dbClient.getWorkspacesV2(locals.user.id, [
    "OWNER",
    "ADMIN",
    "EDITOR",
    "VIEWER",
  ]);
  return json(workspaces);
};

export const POST: RequestHandler = async ({ request, locals }) => {
  const { name, description }: { name: string; description: string } =
    await request.json();

  if (!locals.user || !locals.user.id) {
    return error(401, {
      message: "Authentication required to create workspace",
    });
  }
  if (!name) {
    return error(400, { message: "name is required for workspaces" });
  }

  const workspace = await locals.dbClient.createWorkspace(
    name,
    description,
    locals.user.id,
  );
  return json(workspace);
};
