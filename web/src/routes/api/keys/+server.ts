import { json, error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import { DatabaseClient } from "$lib/server/database";
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
    const { data, error: dbError } = await DatabaseClient.getInstance()
      .from("api_keys")
      .select("id, name, created_at, last_used, revoked")
      .eq("user_id", locals.user.id)
      .eq("revoked", false)
      .order("created_at", { ascending: false });

    if (dbError) {
      console.error("Error fetching API keys:", dbError);
      throw error(500, "Failed to fetch API keys");
    }

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

    const { data: newKeyData, error: insertError } =
      await DatabaseClient.getInstance()
        .from("api_keys")
        .insert({
          key_hash: keyHash,
          name: body.name,
          user_id: locals.user.id,
          revoked: false,
        })
        .select()
        .single();

    if (insertError || !newKeyData) {
      console.error("Error creating API key:", insertError);
      throw error(500, "Failed to create API key");
    }

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
    const { data: keyData, error: fetchError } =
      await DatabaseClient.getInstance()
        .from("api_keys")
        .select("id")
        .eq("id", keyId)
        .eq("user_id", locals.user.id)
        .single();

    if (fetchError || !keyData) {
      throw error(404, "API key not found");
    }

    const { error: updateError } = await DatabaseClient.getInstance()
      .from("api_keys")
      .update({ revoked: true })
      .eq("id", keyId)
      .eq("user_id", locals.user.id);

    if (updateError) {
      throw error(500, "Failed to revoke API key");
    }

    return json({ success: true });
  } catch (err) {
    console.error("Error revoking API key:", err);
    if (err instanceof Error && "status" in err && "body" in err) {
      throw err;
    }
    throw error(500, "Failed to revoke API key");
  }
};
