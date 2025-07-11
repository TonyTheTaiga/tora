import { redirect } from "@sveltejs/kit";

import type { Actions } from "./$types";

export const actions: Actions = {
  default: async ({ locals, cookies }) => {
    cookies.delete("tora_auth_token", { path: "/" });
    redirect(303, "/");
  },
};
