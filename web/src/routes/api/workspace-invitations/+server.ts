import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user || !locals.dbClient || !locals.supabase) {
    return error(401, { message: "Unauthorized" });
  }

  try {
    const { workspaceId, email, roleId } = await request.json();

    if (!workspaceId || !email || !roleId) {
      return error(400, { message: "Missing required fields" });
    }

    const { data: targetUser, error: userError } =
      await locals.adminSupabaseClient.auth.admin.listUsers();

    if (userError) {
      console.error("Error finding user:", userError);
      return error(500, { message: "Error finding user" });
    }

    const user = targetUser.users.find((u) => u.email === email);
    if (!user) {
      return error(404, { message: "User not found" });
    }

    const invitation = await locals.dbClient.createInvitation(
      locals.user.id,
      user.id,
      workspaceId,
      roleId,
    );

    return json(invitation);
  } catch (err) {
    console.error("Error in workspace-invitations API:", err);
    return error(500, { message: "Internal server error" });
  }
};

export const GET: RequestHandler = async ({ locals }) => {
  if (!locals.user || !locals.dbClient) {
    return json([]);
  }

  try {
    const invitations = await locals.dbClient.getPendingInvitationsTo(
      locals.user.id,
      "PENDING",
    );

    return json(invitations);
  } catch (err) {
    console.error("Error in workspace-invitations GET API:", err);
    return json([]);
  }
};
