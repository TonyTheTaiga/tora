import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ locals }) => {
  if (!locals.user || !locals.supabase) {
    return json([]);
  }

  try {
    const { data: roles, error } = await locals.supabase
      .from("workspace_role")
      .select("id, name")
      .order("name");

    if (error) {
      console.error("Error fetching workspace roles:", error);
      return json([]);
    }

    return json(roles || []);
  } catch (err) {
    console.error("Error in workspace-roles API:", err);
    return json([]);
  }
};
