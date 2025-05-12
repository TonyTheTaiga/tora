import { json, error } from "@sveltejs/kit";
import { getExperiments } from "$lib/server/database.js";
import type { SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "$lib/server/database.types";

export async function GET({ url, locals: { supabase } }: { url: URL, locals: { supabase: SupabaseClient<Database> } }) {
  const name_filter = url.searchParams.get("startwith") || "";
  try {
    const experiments = await getExperiments(name_filter);
    return json(experiments);
  } catch (err) {
    if (err instanceof Error) {
      throw error(500, err.message);
    }

    throw error(500, "Internal Error");
  }
}
