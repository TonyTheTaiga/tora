import { redirect } from "@sveltejs/kit";
import type { LayoutServerLoad } from "./$types";

export const load: LayoutServerLoad = async ({ locals, url }) => {
  const isPublicRoute =
    url.pathname === "/" ||
    url.pathname === "/login" ||
    url.pathname === "/docs" ||
    url.pathname.startsWith("/api-docs") ||
    url.pathname.startsWith("/signup");

  if (!locals.session && !isPublicRoute) {
    return redirect(302, "/login");
  }

  if (
    locals.session &&
    (url.pathname === "/login" ||
      url.pathname.startsWith("/signup") ||
      url.pathname === "/")
  ) {
    return redirect(302, "/dashboard");
  }
};
