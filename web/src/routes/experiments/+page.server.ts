import type { Actions } from "./$types";
import type { HyperParam } from "$lib/types";
import { fail } from "@sveltejs/kit";

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
  create: async ({ request, locals }) => {
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
      payload["workspace_id"] = workspaceId;
    }

    await locals.apiClient.post("/api/experiments", payload);
    return { success: true };
  },
  update: async ({ request, locals }) => {
    const {
      "experiment-id": id,
      "experiment-name": name,
      "experiment-description": description,
      tags,
    } = parseFormData(await request.formData());

    await locals.apiClient.put(`/api/experiments/${id}`, {
      "experiment-name": name,
      "experiment-description": description || "",
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
    await locals.apiClient.delete(`/api/experiments/${id}`);

    return { success: true };
  },
};
