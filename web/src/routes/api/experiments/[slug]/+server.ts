import { json } from "@sveltejs/kit";
import { getExperiment, updateExperiment } from "$lib/server/database";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "$lib/server/database.types";

export async function GET({
  params: { slug },
  locals,
}: {
  params: { slug: string };
  locals: { supabase: SupabaseClient<Database>; user: { id: string } | null };
}) {
  // Pass the userId to getExperiment if user is logged in
  const userId = locals.user?.id;
  const experiment = await getExperiment(slug, userId);
  return json(experiment);
}

export async function POST({
  params: { slug },
  request,
}: {
  params: { slug: string };
  request: Request;
}) {
  let data = await request.json();
  await updateExperiment(slug, {
    name: data.name,
    description: data.description,
    tags: data.tags,
    visibility: data.visibility,
  });

  return new Response(
    JSON.stringify({ message: "Experiment updated successfully" }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
}
