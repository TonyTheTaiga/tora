import { error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ params, locals }) => {
  const experimentId = params.slug;
  const userId = locals.user?.id;

  try {
    await locals.dbClient.checkExperimentAccess(experimentId, userId);
  } catch {
    throw error(403, "Access denied");
  }

  const metrics = await locals.dbClient.getMetrics(experimentId);

  const header = "name,value,step,created_at\n";
  const rows = metrics
    .map((m) =>
      [m.name, m.value, m.step ?? "", m.created_at].join(","),
    )
    .join("\n");

  return new Response(header + rows, {
    headers: {
      "Content-Type": "text/csv",
      "Content-Disposition": `attachment; filename=\"${experimentId}-metrics.csv\"`,
    },
  });
};
