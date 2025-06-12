import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { createHash } from "crypto";

function hashApiKey(key: string): string {
  return createHash("sha256").update(key).digest("hex");
}

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
      createdAt: key.createdAt,
      lastUsed: key.lastUsed,
      revoked: key.revoked,
    }));

    return json({ keys });
  } catch (err) {
    console.error("Error fetching API keys:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err;
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

    const prefix = "tora_";
    const randomBytes = Array.from(crypto.getRandomValues(new Uint8Array(24)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

    const fullKey = prefix + randomBytes;
    const keyHash = hashApiKey(fullKey);

    const newKey = await locals.dbClient.createApiKey(
      locals.user.id,
      body.name,
      keyHash,
    );

    return json({ ...newKey, key: fullKey }, { status: 201 });
  } catch (err) {
    console.error("Error creating API key:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err;
    }
    throw error(500, "Failed to create API key");
  }
};
