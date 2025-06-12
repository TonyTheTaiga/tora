import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ params, locals }) => {
  const { session, user } = await locals.safeGetSession();
  if (!user) {
    throw error(401, "Authentication required");
  }

  const workspaceId = params.id;

  try {
    // For now, return mock data
    // In a real implementation, this would fetch actual workspace members
    const members = [
      {
        id: user.id,
        email: user.email,
        role: "OWNER",
        joinedAt: new Date().toISOString()
      }
    ];

    return json({
      success: true,
      members: members,
    });
  } catch (err) {
    console.error("Error getting workspace members:", err);
    throw error(500, "Failed to get workspace members");
  }
};