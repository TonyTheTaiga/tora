import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import type { WorkspaceRole } from "$lib/types";

export const POST: RequestHandler = async ({ params, request, locals }) => {
  const { session, user } = await locals.safeGetSession();
  if (!user) {
    throw error(401, "Authentication required");
  }

  const workspaceSlug = params.slug;
  const { email, role }: { email: string; role: WorkspaceRole } =
    await request.json();

  if (!email || !role) {
    throw error(400, "Email and role are required");
  }

  // Validate role
  const validRoles: WorkspaceRole[] = ["VIEWER", "EDITOR", "ADMIN"];
  if (!validRoles.includes(role)) {
    throw error(400, "Invalid role");
  }

  try {
    // For now, we'll create a simple invitation
    // In a real implementation, this would send an email and create a proper invitation flow
    return json({
      success: true,
      message: `Invitation would be sent to ${email} for role ${role}`,
      invitation: {
        workspace_slug: workspaceSlug,
        email: email,
        role: role,
        created_at: new Date().toISOString(),
      },
    });
  } catch (err) {
    console.error("Error inviting user to workspace:", err);
    throw error(500, "Failed to send workspace invitation");
  }
};

// Get pending invitations for a workspace
export const GET: RequestHandler = async ({ params, locals }) => {
  const { session, user } = await locals.safeGetSession();
  if (!user) {
    throw error(401, "Authentication required");
  }

  const workspaceSlug = params.slug;

  try {
    // This would get pending invitations from the database
    // Using the methods you added to the dbClient
    const pendingInvitations = await locals.dbClient.getPendingInvitationsFrom(
      user.email!,
      "pending",
    );

    return json({
      success: true,
      invitations: pendingInvitations,
    });
  } catch (err) {
    console.error("Error getting pending invitations:", err);
    return json({
      success: true,
      invitations: [],
    });
  }
};