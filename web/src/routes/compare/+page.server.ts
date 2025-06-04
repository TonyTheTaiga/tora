import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ url, locals }) => {
  const idsParam = url.searchParams.get("ids");
  if (!idsParam) {
    throw error(400, "Missing 'ids' query parameter");
  }
  const ids = idsParam.split(",").filter(Boolean);
  const { data: experiments, error: supError } = await locals.supabase.rpc(
    "get_experiments_with_metrics_names",
    { experiment_ids: ids },
  );

  if (supError) {
    console.log(supError);
    throw error(400, "Failed to get experiments");
  }

  return { experiments };
};
