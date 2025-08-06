import type { Actions } from "@sveltejs/kit";
import { fail } from "@sveltejs/kit";
import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { ApiResponse, Workspace, HyperParam } from "$lib/types";

interface FormDataResult {
  hyperparams: HyperParam[];
  tags: string[];
  [key: string]: any;
}

function parseFormData(formData: FormData): FormDataResult {
  const data = Object.fromEntries(formData);
  const result: FormDataResult = {
    hyperparams: [],
    tags: [],
  };
  const hyperparamMap = new Map<number, Partial<HyperParam>>();

  for (const [key, value] of Object.entries(data)) {
    if (typeof value !== "string") continue;

    if (key.startsWith("hyperparams.")) {
      const [, indexStr, field] = key.split(".");
      const index = Number(indexStr);
      const existing = hyperparamMap.get(index) ?? {};
      hyperparamMap.set(index, { ...existing, [field]: value });
    } else if (key.startsWith("tags.")) {
      const [, indexStr] = key.split(".");
      result.tags[Number(indexStr)] = value;
    } else {
      result[key] = value;
    }
  }

  result.hyperparams = [...hyperparamMap.values()].filter(
    (hp): hp is HyperParam => hp.key != null && hp.value != null,
  );
  result.tags = result.tags.filter(Boolean);

  return result;
}

export const actions: Actions = {
  createWorkspace: async ({ request, locals }) => {
    if (!locals.session) {
      return fail(401, { error: "Authentication required" });
    }

    if (!locals.apiClient) {
      return fail(500, { error: "API client not available" });
    }

    const requestId = generateRequestId();
    const timer = startTimer("workspace.create", {
      requestId,
      userId: locals.session.user.id,
    });

    try {
      const formData = await request.formData();
      const name = formData.get("name") as string;
      const description = formData.get("description") as string;

      if (!name || name.trim().length === 0) {
        timer.end({ error: "Invalid name" });
        return fail(400, { error: "Workspace name is required" });
      }

      const response = await locals.apiClient.post<ApiResponse<Workspace>>(
        "/api/workspaces",
        {
          name: name.trim(),
          description: description?.trim() || null,
        },
      );

      if (response.status !== 201) {
        return fail(500, { error: "Failed to create workspace" });
      }

      const workspace = response.data;

      timer.end({
        workspaceId: workspace.id,
        workspaceName: workspace.name,
      });

      return {
        id: workspace.id,
        name: workspace.name,
        description: workspace.description,
        createdAt: new Date(workspace.createdAt),
        role: workspace.role,
      };
    } catch (err) {
      timer.end({
        error: err instanceof Error ? err.message : "Unknown error",
      });

      console.error("Error creating workspace:", err);

      if (err instanceof Error && err.message.includes("401")) {
        return fail(401, { error: "Authentication required" });
      }

      return fail(500, {
        error: "Failed to create workspace. Please try again.",
      });
    }
  },

  deleteWorkspace: async ({ request, locals, fetch }) => {
    if (!locals.session) {
      return fail(401, { error: "Authentication required" });
    }

    if (!locals.apiClient) {
      return fail(500, { error: "API client not available" });
    }

    const requestId = generateRequestId();
    const timer = startTimer("workspace.create", {
      requestId,
      userId: locals.session.user.id,
    });

    try {
      const data = await request.formData();
      const workspaceId = data.get("workspaceId") as string;
      const response = await fetch(`/api/workspaces/${workspaceId}`, {
        method: "DELETE",
      });
      return { success: true };
    } catch (err) {
      console.error("failed to delete workspace", err);
      fail(500, { error: `failed to delete workspace ${err}` });
    }

    timer.end({});
  },

  sendInvitation: async ({ request, locals }) => {
    const data = await request.formData();
    const workspaceId = data.get("workspaceId") as string;
    const email = data.get("email") as string;
    const roleId = data.get("roleId") as string;

    if (!locals.apiClient.hasElevatedPermissions()) {
      return fail(401, { error: "Authentication required" });
    }

    try {
      await locals.apiClient.post("/api/workspace-invitations", {
        workspaceId,
        email,
        roleId,
      });
      return { success: true };
    } catch (error) {
      console.error("Failed to send invitation:", error);
      return fail(500, { error: "Failed to send invitation" });
    }
  },

  createExperiment: async ({ request, locals }) => {
    const {
      "experiment-name": name,
      "experiment-description": description,
      "workspace-id": workspaceId,
      tags,
    } = parseFormData(await request.formData());

    const payload: any = {
      name: name,
      description: description || "",
      tags,
    };

    if (workspaceId) {
      payload["workspaceId"] = workspaceId;
    }

    await locals.apiClient.post("/api/experiments", payload);
    return { success: true };
  },
  updateExperiment: async ({ request, locals }) => {
    const {
      "experiment-id": id,
      "experiment-name": name,
      "experiment-description": description,
      tags,
    } = parseFormData(await request.formData());
    await locals.apiClient.put(`/api/experiments/${id}`, {
      name: name,
      description: description || "",
      tags,
    });

    return { success: true };
  },
  delete: async ({ request, locals }) => {
    const data = await request.formData();
    const id = data.get("id");

    if (!id || typeof id !== "string") {
      return fail(400, {
        message: "A valid ID is required",
      });
    }
    await locals.apiClient.delete<null>(`/api/experiments/${id}`);
    return { success: true };
  },
};
