import { redirect } from "@sveltejs/kit";
import type { LayoutServerLoad } from "./$types";

export const load: LayoutServerLoad = async ({ locals, url }) => {
  if (locals.session && url.pathname === "/") {
    throw redirect(302, "/workspaces");
  }
};
