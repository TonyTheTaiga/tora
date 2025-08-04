import { redirect } from "@sveltejs/kit";
import type { LayoutServerLoad } from "./$types";

export const load: LayoutServerLoad = async ({ locals, url }) => {
  if (
    !locals.session &&
    url.pathname !== "/login" &&
    !url.pathname.startsWith("/signup") &&
    url.pathname !== "/"
  ) {
    return redirect(302, "/login");
  } else if (
    locals.session &&
    (url.pathname === "/login" ||
      url.pathname.startsWith("/signup") ||
      url.pathname === "/")
  ) {
    return redirect(302, "/dashboard");
  }
};
