import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";

export const load: PageServerLoad = async ({ url, locals }) => {
  const idsParam = url.searchParams.get("ids");
  if (!idsParam) {
    throw error(400, "Missing 'ids' query parameter");
  }
  const ids = idsParam.split(",").filter(Boolean);
  const { data, error: supError } = await locals.supabase.rpc(
    "get_experiments_and_metrics",
    { experiment_ids: ids },
  );

  if (supError) {
    console.log(supError);
    throw error(400, "Failed to get experiments");
  }

  const experiments = data
    .map((item) => ({
      id: item.id,
      name: item.name,
      visibility: item.visibility,
      description: item.description,
      availableMetrics: item.metric_dict ? Object.keys(item.metric_dict) : [],
      metricData: item.metric_dict,
      tags: item.tags,
      hyperparams: item.hyperparams
        ? (item.hyperparams.map((hp: any) => ({
            key: hp.name || hp.key,
            value: hp.value,
          })) as HyperParam[])
        : null,
      createdAt: new Date(item.created_at),
    }))
    .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());

  return { experiments };
};
