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
  if (!user) {
    return error(500, {
      message: "Cannot create a experiment for anonymous user",
    });
  }
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

  const workspace = await createWorkspace(rawName.trim(), description, user.id);
  return json(workspace);
}
