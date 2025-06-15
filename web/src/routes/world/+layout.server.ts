import type { LayoutServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";

export const load: LayoutServerLoad = async ({ locals }) => {
  const requestId = generateRequestId();
  const timer = startTimer("world", { requestId });
  try {
    const experiments = await locals.dbClient.getPublicExperiments();
    timer.end({});

    return {
      experiments,
    };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
