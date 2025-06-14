import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import type { WorkspaceRole } from "$lib/types";

export const PATCH: RequestHandler = async ({ params, request, locals }) => {
  const { session, user } = await locals.safeGetSession();
  if (!user) {
    throw error(401, "Authentication required");
  }

  const workspaceSlug = params.slug;
  const memberId = params.memberId;
  const { role }: { role: WorkspaceRole } = await request.json();

  if (!role) {
    throw error(400, "Role is required");
  }

  try {
    // In a real implementation, this would:
    // 1. Verify user has OWNER permissions for the workspace
    // 2. Update the member's role in the database
    // 3. Return the updated member data

    return json({
      success: true,
      message: `Member role updated to ${role}`,
    });
  } catch (err) {
    console.error("Error updating member role:", err);
    throw error(500, "Failed to update member role");
  }
};

export const DELETE: RequestHandler = async ({ params, locals }) => {
  const { session, user } = await locals.safeGetSession();
  if (!user) {
    throw error(401, "Authentication required");
  }

  const workspaceSlug = params.slug;
  const memberId = params.memberId;

  if (!workspaceSlug || !memberId) {
    throw error(400, "Workspace slug and member ID are required");
  }

  try {
    const db = locals.dbClient;
    // The slug parameter is actually the workspace ID (UUID) in this implementation
    await db.removeWorkspaceRole(workspaceSlug, memberId);

    return json({
      success: true,
      message: "Member removed from workspace",
    });
  } catch (err) {
    console.error("Error removing member:", err);
    throw error(500, "Failed to remove member");
  }
};
