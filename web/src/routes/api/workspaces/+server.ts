import { json, error } from "@sveltejs/kit";
import { getWorkspaces, createWorkspace } from "$lib/server/database";

export async function GET({ locals: { user } }) {
  if (!user) {
    return json([]);
  }

  const workspaces = await getWorkspaces(user.id);
  return json(workspaces);
}

export async function POST({ request, locals: { user } }) {
  const { name, description }: { name: string; description: string } =
    await request.json();

  if (!user) {
    return error(500, {
      message: "Cannot create a experiment for anonymous user",
    });
  }
  if (!name) {
    return error(500, { message: "name is required for workspaces" });
  }
  const workspace = await createWorkspace(name, description, user.id);
  return json(workspace);
}
