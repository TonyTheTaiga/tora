import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ fetch, locals, parent, url }) => {
  const requestId = generateRequestId();
  const timer = startTimer("home.load", {
    requestId,
  });
  try {
    timer.end({});
    return {};
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
