import type { LayoutServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import { error } from "@sveltejs/kit";

export const load: LayoutServerLoad = async ({ locals }) => {
  if (!locals.user) {
    error(501, "user required");
  }
  const requestId = generateRequestId();
  const timer = startTimer("workspaces.load", {
    requestId,
  });

  try {
    const workspaces = await locals.dbClient.getWorkspacesV2(locals.user.id, [
      "OWNER",
      "ADMIN",
      "EDITOR",
      "VIEWER",
    ]);
    timer.end({ user_id: locals.user.id });
    return { workspaces };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};
