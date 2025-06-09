import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { createHash } from "crypto";

// Helper to hash an API key for storage
function hashApiKey(key: string): string {
  return createHash("sha256").update(key).digest("hex");
}

// GET /api/keys - Get all API keys for the authenticated user
export const GET: RequestHandler = async ({ locals }) => {
  if (!locals.user) {
    throw error(401, "Unauthorized");
  }

  try {
    const data = await locals.dbClient.getApiKeys(locals.user.id);

    const keys = data.map((key) => ({
      id: key.id,
      prefix: "tosk_",
      name: key.name,
      createdAt: key.created_at,
      lastUsed: key.last_used,
      revoked: key.revoked,
    }));

    return json({ keys });
  } catch (err) {
    console.error("Error fetching API keys:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err; // Re-throw SvelteKit errors
    }
    throw error(500, "Failed to fetch API keys");
  }
};

export const POST: RequestHandler = async ({ request, locals }) => {
  if (!locals.user) {
    throw error(401, "Unauthorized");
  }

  try {
    const body = await request.json();

    if (
      !body.name ||
      typeof body.name !== "string" ||
      body.name.trim() === ""
    ) {
      throw error(400, "API key name is required");
    }

    const prefix = "tosk_";
    const randomBytes = Array.from(crypto.getRandomValues(new Uint8Array(24)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    const fullKey = prefix + randomBytes;

    const keyHash = hashApiKey(fullKey);

    const newKeyData = await locals.dbClient.createApiKey(
      locals.user.id,
      body.name,
      keyHash,
    );

    const newKey = {
      id: newKeyData.id,
      name: body.name,
      prefix,
      key: fullKey,
      createdAt: newKeyData.created_at,
      revoked: newKeyData.revoked,
    };

    return json({ key: newKey }, { status: 201 });
  } catch (err) {
    console.error("Error creating API key:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err;
    }
    throw error(500, "Failed to create API key");
  }
};

export const DELETE: RequestHandler = async ({ url, locals }) => {
  if (!locals.user) {
    throw error(401, "Unauthorized");
  }

  const keyId = url.searchParams.get("id");
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
