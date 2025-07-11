import { error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const GET: RequestHandler = async ({ params, locals }) => {
  if (!locals.session) {
    error(401, "Authentication required");
  }

  if (!locals.apiClient) {
    error(500, "API client not available");
  }

  const experimentId = params.experimentId;

  try {
    // Assuming your Rust backend has an endpoint like /api/experiments/{id}/metrics/csv
    // that returns the CSV content directly.
    const response = await locals.apiClient.request<Response>(
      `/api/experiments/${experimentId}/metrics/csv`,
      {
        method: "GET",
        headers: {
          Accept: "text/csv",
        },
      },
    );

    const csvContent = await response.text(); // Assuming it's text/csv

    return new Response(csvContent, {
      headers: {
        "Content-Type": "text/csv",
        "Content-Disposition": `attachment; filename="experiment_${experimentId}_metrics.csv"`,
      },
    });
  } catch (err) {
    console.error(`Error fetching CSV for experiment ${experimentId}:`, err);
    error(500, "Failed to fetch CSV data");
  }
};
