import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const PATCH: RequestHandler = async ({ request, locals, url }) => {
  if (!locals.user || !locals.dbClient) {
    return error(401, { message: "Unauthorized" });
  }

  try {
    const invitationId = url.searchParams.get("invitationId");
    const action = url.searchParams.get("action");

    if (!invitationId || !action) {
      return error(400, {
        message: "Missing invitationId or action parameter",
      });
    }

    if (action !== "accept" && action !== "deny") {
      return error(400, { message: "Action must be 'accept' or 'deny'" });
    }

    if (action === "accept") {
      // First get the invitation details
      const { data: invitation, error: inviteError } = await locals.supabase
        .from("workspace_invitations")
        .select("workspace_id, role_id, to")
        .eq("id", invitationId)
        .eq("to", locals.user.id)
        .single();

      if (inviteError || !invitation) {
        return error(404, { message: "Invitation not found" });
      }

      // Add user to the workspace
      const { error: workspaceError } = await locals.supabase
        .from("user_workspaces")
        .insert({
          user_id: locals.user.id,
          workspace_id: invitation.workspace_id,
          role_id: invitation.role_id,
        });

      if (workspaceError) {
        console.error("Error adding user to workspace:", workspaceError);
        return error(500, { message: "Failed to add user to workspace" });
      }

      // Mark invitation as accepted
      await locals.dbClient.markInvitationAsAccepted(invitationId);
    } else {
      // Mark invitation as denied
      const { error: denyError } = await locals.supabase
        .from("workspace_invitations")
        .update({ status: "denied" })
        .eq("id", invitationId)
        .eq("to", locals.user.id);

      if (denyError) {
        console.error("Error denying invitation:", denyError);
        return error(500, { message: "Failed to deny invitation" });
      }
    }

    return json({ success: true });
  } catch (err) {
    console.error("Error handling invitation response:", err);
    return error(500, { message: "Internal server error" });
  }
};
