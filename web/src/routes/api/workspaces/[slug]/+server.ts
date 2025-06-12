import type { RequestHandler } from "./$types";
import { json } from "@sveltejs/kit";

export const DELETE: RequestHandler = async ({ locals, params }) => {
  const slug = params.slug;
  await locals.dbClient.deleteWorkspace(slug);
  return json({ status: 200 });
};
