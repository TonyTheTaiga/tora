import { json, error } from "@sveltejs/kit";
import { DatabaseClient } from "$lib/server/database";

export async function GET() {
  const workspaces = await DatabaseClient.getWorkspaces();
  return json(workspaces);

}
