import type { RequestHandler } from "./$types";
import { error, json } from "@sveltejs/kit";

export const DELETE: RequestHandler = async ({ locals, params }) => {
  if (!locals.user) {
    throw error(401, "Unauthorized");
  }

  const keyId = params.slug;
  if (!keyId) {
    throw error(400, "Key ID is required");
  }

  try {
    await locals.dbClient.revokeApiKey(locals.user.id, keyId);
    return json({ success: true });
  } catch (err) {
    console.error("Error revoking API key:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err;
    }
    throw error(500, "Failed to revoke API key");
  }
};
