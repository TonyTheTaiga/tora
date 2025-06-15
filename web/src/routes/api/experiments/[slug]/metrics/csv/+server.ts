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

  const names = Array.from(new Set(metrics.map((m) => m.name))).sort();
  const steps = Array.from(new Set(metrics.map((m) => m.step ?? 0))).sort(
    (a, b) => a - b,
  );

  const data = new Map<number, Record<string, number>>();
  for (const m of metrics) {
    const step = m.step ?? 0;
    if (!data.has(step)) data.set(step, {});
    data.get(step)![m.name] = m.value;
  }

  const header = ["step", ...names].join(",") + "\n";
  const rows = steps
    .map((step) =>
      [step, ...names.map((n) => data.get(step)?.[n] ?? "")].join(","),
    )
    .join("\n");

  return new Response(header + rows, {
    headers: {
      "Content-Type": "text/csv",
      "Content-Disposition": `attachment; filename=\"${experimentId}-metrics.csv\"`,
    },
  });
};
