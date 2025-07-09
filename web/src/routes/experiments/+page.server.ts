import type { Actions, PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";
import { error, fail } from "@sveltejs/kit";
import { generateRequestId, startTimer } from "$lib/utils/timing";

interface ApiResponse<T> {
  status: number;
  data: T;
}

interface ExperimentData {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  hyperparams: any[];
  workspace_id: string;
  available_metrics: string[];
}

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

export const load: PageServerLoad = async ({ locals, url }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("experiments.load", { requestId });

  try {
    const workspace = url.searchParams.get("workspace");
    const apiUrl = workspace
      ? `/api/experiments?workspace=${workspace}`
      : "/api/experiments";

    const experimentsResponse =
      await locals.apiClient.get<ApiResponse<ExperimentData[]>>(apiUrl);

    if (experimentsResponse.status !== 200) {
      error(500, "Failed to fetch experiments");
    }

    const rawExperiments = experimentsResponse.data || [];
    const experiments = rawExperiments.map((exp) => ({
      id: exp.id,
      name: exp.name,
      description: exp.description || "",
      hyperparams: exp.hyperparams || [],
      tags: exp.tags || [],
      createdAt: new Date(exp.created_at),
      updatedAt: new Date(exp.updated_at),
      availableMetrics: exp.available_metrics || [],
      workspaceId: exp.workspace_id,
    }));

    timer.end({
      experiments_count: experiments.length,
      user_id: locals.session.user.id,
      workspace_filter: workspace || "all",
    });

    return {
      experiments,
      workspace,
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });

    console.error("Error loading experiments:", err);

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load experiments");
  }
};

export const actions: Actions = {
  create: async ({ request, locals }) => {
    if (!locals.session) {
      return fail(401, {
        message: "Authentication required",
      });
    }

    if (!locals.apiClient) {
      return fail(500, {
        message: "API client not available",
      });
    }

    try {
      const {
        "experiment-name": name,
        "experiment-description": description,
        "workspace-id": workspaceId,
        tags,
      } = parseFormData(await request.formData());

      if (!name || typeof name !== "string") {
        return fail(400, {
          message: "Experiment name is required",
        });
      }

      if (!workspaceId || typeof workspaceId !== "string") {
        return fail(400, {
          message: "Workspace ID is required",
        });
      }

      const createResponse = await locals.apiClient.post<
        ApiResponse<ExperimentData>
      >("/api/experiments", {
        "experiment-name": name,
        "experiment-description": description || "",
        "workspace-id": workspaceId,
        tags: tags || [],
      });

      if (createResponse.status !== 201) {
        return fail(createResponse.status, {
          message: "Failed to create experiment",
        });
      }

      return { success: true };
    } catch (err) {
      console.error("Error creating experiment:", err);
      return fail(500, {
        message: "Failed to create experiment",
      });
    }
  },

  update: async ({ request, locals }) => {
    if (!locals.session) {
      return fail(401, {
        message: "Authentication required",
      });
    }

    if (!locals.apiClient) {
      return fail(500, {
        message: "API client not available",
      });
    }

    try {
      const {
        "experiment-id": id,
        "experiment-name": name,
        "experiment-description": description,
        tags,
      } = parseFormData(await request.formData());

      if (!id || typeof id !== "string") {
        return fail(400, {
          message: "Experiment ID is required",
        });
      }

      if (!name || typeof name !== "string") {
        return fail(400, {
          message: "Experiment name is required",
        });
      }

      const updateResponse = await locals.apiClient.put<
        ApiResponse<ExperimentData>
      >(`/api/experiments/${id}`, {
        "experiment-id": id,
        "experiment-name": name,
        "experiment-description": description || "",
        tags: tags || [],
      });

      if (updateResponse.status !== 200) {
        return fail(updateResponse.status, {
          message: "Failed to update experiment",
        });
      }

      return { success: true };
    } catch (err) {
      console.error("Error updating experiment:", err);
      return fail(500, {
        message: "Failed to update experiment",
      });
    }
  },

  delete: async ({ request, locals }) => {
    if (!locals.session) {
      return fail(401, {
        message: "Authentication required",
      });
    }

    if (!locals.apiClient) {
      return fail(500, {
        message: "API client not available",
      });
    }

    try {
      const data = await request.formData();
      const id = data.get("id");

      if (!id || typeof id !== "string") {
        return fail(400, {
          message: "A valid ID is required",
        });
      }

      const deleteResponse = await locals.apiClient.delete<ApiResponse<void>>(
        `/api/experiments/${id}`,
      );

      if (deleteResponse.status !== 204) {
        return fail(deleteResponse.status, {
          message: "Failed to delete experiment",
        });
      }

      return { success: true };
    } catch (err) {
      console.error("Error deleting experiment:", err);
      return fail(500, {
        message: "Failed to delete experiment",
      });
    }
  },
};
