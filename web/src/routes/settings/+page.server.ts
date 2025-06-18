import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { PageServerLoad, Actions } from "./$types";
import { error, redirect } from "@sveltejs/kit";
import type { ApiKey } from "$lib/types";
import { createHash } from "crypto";

export const load: PageServerLoad = async ({ locals }) => {
  const { session, user } = await locals.safeGetSession();

  const requestId = generateRequestId();
  const timer = startTimer("settings.load", { requestId });

  if (!user) {
    const experiments = new Array();
    return { experiments, session };
  }

  const workspaces = await locals.dbClient.getWorkspacesV2(user.id, [
    "OWNER",
    "ADMIN",
    "EDITOR",
    "VIEWER",
  ]);

  let apiKeys: ApiKey[] = [];
  try {
    const data = await locals.dbClient.getApiKeys(user.id);
    apiKeys = data.map((key) => ({
      id: key.id,
      prefix: "tosk_",
      name: key.name,
      createdAt: key.createdAt,
      lastUsed: key.lastUsed,
      revoked: key.revoked,
    }));
  } catch (err) {
    console.error("Error fetching API keys:", err);
  }

  timer.end({});
  return { workspaces, apiKeys, session };
};

export const actions: Actions = {
  createWorkspace: async ({ request, locals }) => {
    const fd = await request.formData();
    const rawName = fd.get("name");
    const rawDescription = fd.get("description");

    if (typeof rawName !== "string" || !rawName.trim()) {
      throw error(400, { message: "name required" });
    }
    const description =
      typeof rawDescription === "string" && rawDescription.trim()
        ? rawDescription
        : null;

    if (!locals.user || !locals.user.id) {
      throw error(401, "Authentication required to create workspace");
    }

    const workspace = await locals.dbClient.createWorkspace(
      rawName.trim(),
      description,
      locals.user.id,
    );
    return workspace;
  },

  deleteWorkspace: async ({ request, locals }) => {
    const formData = await request.formData();
    const { id } = Object.fromEntries(formData.entries());
    if (!id) {
      throw error(400, "ID required to delete workspace");
    }
    await locals.dbClient.deleteWorkspace(id as string);

    return redirect(300, "/settings");
  },

  removeSharedWorkspace: async ({ request, locals }) => {
    const formData = await request.formData();
    const { userId, workspaceId } = Object.fromEntries(formData.entries());

    if (!(userId && workspaceId)) {
      throw error(
        400,
        "userId and workspaceId required to remove shared workspace",
      );
    }

    try {
      await locals.dbClient.removeWorkspaceRole(
        workspaceId as string,
        userId as string,
      );
    } catch (err) {
      console.error("Error removing member:", err);
      throw error(500, "Failed to leave workspace");
    }

    return redirect(300, "/settings");
  },

  createApiKey: async ({ request, locals }) => {
    const formData = await request.formData();
    const { name } = Object.fromEntries(formData.entries());
    if (!name) {
      throw error(400, "A unique name is required for your api key");
    }

    if (!locals.user) {
      throw error(401, "Unauthorized");
    }

    const prefix = "tora_";
    const randomBytes = Array.from(crypto.getRandomValues(new Uint8Array(24)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

    const fullKey = prefix + randomBytes;
    const keyHash = createHash("sha256").update(fullKey).digest("hex");

    const newKey = await locals.dbClient.createApiKey(
      locals.user.id,
      name as string,
      keyHash,
    );

    return { ...newKey, key: fullKey };
  },

  revokeApiKey: async ({ request, locals }) => {
    const formData = await request.formData();
    const { id } = Object.fromEntries(formData.entries());
    if (!id) {
      throw error(400, "ID reuired to revoke api key");
    }

    if (!locals.user) {
      throw error(401, "Unauthorized");
    }

    await locals.dbClient.revokeApiKey(locals.user.id, id as string);
    return redirect(300, "/settings");
  },
};
