import type { PageServerLoad } from "./$types";
import { error } from "@sveltejs/kit";
import { startTimer, generateRequestId } from "$lib/utils/timing";

interface ApiResponse<T> {
  status: number;
  data: T;
}

interface WorkspaceData {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  role: string;
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
}

export const load: PageServerLoad = async ({ locals, params, parent }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const requestId = generateRequestId();
  const timer = startTimer("workspace.load", { requestId });

  const workspaceId = params.slug;

  try {
    const { workspaces } = await parent();
    const currentWorkspace = workspaces.find((w: any) => w.id === workspaceId);
    if (!currentWorkspace) {
      error(404, "Workspace not found");
    }

    let experiments: any[] = [];
    try {
      const experimentsResponse = await locals.apiClient.get<
        ApiResponse<ExperimentData[]>
      >(`/api/workspaces/${workspaceId}/experiments`);

      if (experimentsResponse.status === 200) {
        const rawExperiments = experimentsResponse.data || [];
        experiments = rawExperiments.map((exp) => ({
          id: exp.id,
          name: exp.name,
          description: exp.description || "",
          hyperparams: exp.hyperparams || [],
          tags: exp.tags || [],
          createdAt: new Date(exp.created_at),
          updatedAt: new Date(exp.updated_at),
          availableMetrics: [], // TODO: Implement when metrics API is ready
          workspaceId: workspaceId,
        }));
      }
    } catch (expErr) {
      console.error("Error fetching experiments:", expErr);
    }

    timer.end({
      workspace_id: workspaceId,
      workspace_name: currentWorkspace.name,
      experiments_count: experiments.length,
    });

    return {
      currentWorkspace,
      experiments,
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });

    console.error("Error loading workspace:", err);

    if (err instanceof Error && err.message.includes("404")) {
      error(404, "Workspace not found");
    }

    if (err instanceof Error && err.message.includes("401")) {
      error(401, "Authentication required");
    }

    error(500, "Failed to load workspace");
  }
};
