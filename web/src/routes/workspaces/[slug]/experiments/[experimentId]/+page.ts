import { error } from "@sveltejs/kit";
import type { PageLoad } from "./$types";

export const load: PageLoad = async ({ params, fetch, parent }) => {
  const parentData = await parent();
  const workspace = parentData.workspaces?.find((w) => w.id === params.slug);

  try {
    // Load experiment details
    const experimentResponse = await fetch(
      `/api/experiments/${params.experimentId}`,
    );
    if (!experimentResponse.ok) {
      throw error(404, "Experiment not found");
    }
    const experiment = await experimentResponse.json();

    // Load metrics for the experiment
    let metrics = [];
    try {
      const metricsResponse = await fetch(
        `/api/experiments/${params.experimentId}/metrics`,
      );
      if (metricsResponse.ok) {
        metrics = await metricsResponse.json();
      }
    } catch (err) {
      console.warn("Failed to load metrics:", err);
    }

    // Load experiment files/artifacts if available
    let files = [];
    try {
      const filesResponse = await fetch(
        `/api/experiments/${params.experimentId}/files`,
      );
      if (filesResponse.ok) {
        files = await filesResponse.json();
      }
    } catch (err) {
      console.warn("Failed to load files:", err);
    }

    return {
      experiment,
      metrics,
      files,
      workspace,
    };
  } catch (err) {
    throw error(500, "Failed to load experiment data");
  }
};
