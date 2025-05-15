import { json, error } from "@sveltejs/kit";
import { getExperiments, createExperiment } from "$lib/server/database";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "$lib/server/database.types";

export async function GET({ url, locals }: { url: URL, locals: { supabase: SupabaseClient<Database>, user: { id: string } | null } }) {
  const name_filter = url.searchParams.get("startwith") || "";
  try {
    const userId = locals.user?.id;
    const experiments = await getExperiments(name_filter, userId);
    return json(experiments);
  } catch (err) {
    if (err instanceof Error) {
      throw error(500, err.message);
    }

    throw error(500, "Internal Error");
  }
}

export async function POST({ request, locals: { user } }) {
  if (!user) {
    return json({ error: "Cannot create a experiment for anonymous user" }, { status: 500 });
  }

  try {
    let data = await request.json();
    let name = data["name"];
    let description = data["description"];
    let hyperparams = data["hyperparams"];
    let tags = data["tags"];
    let visibility = data["visibility"] || "PRIVATE";

    if (typeof hyperparams === "string") {
      try {
        hyperparams = JSON.parse(hyperparams);
      } catch (e) {
        console.error("Failed to parse hyperparams:", e);
        return json({ error: "Invalid hyperparams format" }, { status: 400 });
      }
    }
    const experiment = await createExperiment(
      user.id,
      name,
      description,
      hyperparams,
      tags,
      visibility,
    );
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
}
