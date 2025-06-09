import { error } from "@sveltejs/kit";
import type { PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";

export interface ExperimentWithMetrics {
  id: string;
  name: string;
  visibility: string;
  description: string;
  metricData: Record<string, number[]>;
  tags: string[];
  hyperparams: HyperParam[] | null;
  createdAt: Date;
}

export const load: PageServerLoad = async ({ url, locals }) => {
  const idsParam = url.searchParams.get("ids");
  if (!idsParam) {
    throw error(400, "Missing 'ids' query parameter");
  }
  const ids = idsParam.split(",").filter(Boolean);
  const data = await locals.dbClient.getExperimentsAndMetrics(ids);

  const experiments = data
    .map((item) => ({
      id: item.id,
      name: item.name,
      visibility: item.visibility,
      description: item.description,
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
    .sort(
      (a, b) => b.createdAt.getTime() - a.createdAt.getTime(),
    ) as ExperimentWithMetrics[];

  return { experiments };
};
