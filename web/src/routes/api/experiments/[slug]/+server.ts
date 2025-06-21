import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ params, locals }) => {
  const experiment = await locals.dbClient.getExperiment(params.slug);
  return json(experiment);
};

export const POST: RequestHandler = async ({ params, request, locals }) => {
  let data = await request.json();
  await locals.dbClient.updateExperiment(params.slug, {
    name: data.name,
    description: data.description,
    tags: data.tags,
  });

  return new Response(
    JSON.stringify({ message: "Experiment updated successfully" }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
};

export const DELETE: RequestHandler = async ({ params, locals }) => {
  await locals.dbClient.deleteExperiment(params.slug);

  return new Response(
    JSON.stringify({ message: "Experiment deleted successfully" }),
    {
      status: 200,
      headers: { "Content-Type": "application/json" },
    },
  );
};
