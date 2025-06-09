import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ url, locals }) => {
  const { user } = locals;
  if (!user) {
    throw error(401, "Unauthorized");
  }

  const workspace_id = url.searchParams.get("workspace") || undefined;

  try {
    const experiments = await locals.dbClient.getExperiments(
      user.id,
      workspace_id,
    );
    return json(experiments);
  } catch (err) {
    const errorMessage =
      err instanceof Error ? err.message : "Internal Server Error";
    throw error(500, errorMessage);
  }
};

export const POST: RequestHandler = async ({ request, locals, cookies }) => {
  if (!locals.user) {
    return json(
      { error: "Cannot create a experiment for anonymous user" },
      { status: 500 },
    );
  }

  try {
    let data = await request.json();
    let name = data["name"];
    let description = data["description"];
    let hyperparams = data["hyperparams"];
    let tags = data["tags"];
    let visibility = data["visibility"] || "PRIVATE";
    let workspaceId = data["workspaceId"] || cookies.get("current_workspace");

    if (workspaceId === "API_DEFAULT") {
      const workspace = await locals.dbClient.getOrCreateDefaultWorkspace(
        locals.user.id,
      );
      workspaceId = workspace.id;
    }

    if (typeof hyperparams === "string") {
      try {
        hyperparams = JSON.parse(hyperparams);
      } catch (e) {
        console.error("Failed to parse hyperparams:", e);
        return json({ error: "Invalid hyperparams format" }, { status: 400 });
      }
    }
    const experiment = await locals.dbClient.createExperiment(locals.user.id, {
      name,
      description,
      hyperparams,
      tags,
      visibility,
      workspaceId,
    });
    return json({ success: true, experiment: experiment });
  } catch (error: unknown) {
    return json(
      {
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 },
    );
  }
};
