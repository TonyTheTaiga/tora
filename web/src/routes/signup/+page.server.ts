import { redirect, fail } from "@sveltejs/kit";
import type { Actions } from "./$types";
import { dev } from "$app/environment";
import type { SessionData } from "$lib/types";

type LoginResponse = {
  status: number;
  data: SessionData;
};

export const actions: Actions = {
  default: async ({ request, cookies, locals }) => {
    const data = await request.formData();
    const email = data.get("email") as string;
    const password = data.get("password") as string;

    if (!email || !password) {
      return fail(400, {
        error: "Email and password are required",
      });
    }

    try {
      await locals.apiClient.post("/signup", {
        email,
        password,
      });

      const response = await locals.apiClient.post<LoginResponse>("/login", {
        email,
        password,
      });

      const sessionData: SessionData = {
        access_token: response.data.access_token,
        refresh_token: response.data.refresh_token,
        expires_in: response.data.expires_in,
        expires_at: response.data.expires_at,
        user: {
          id: response.data.user.id,
          email: response.data.user.email,
        },
      };

      const sessionJson = JSON.stringify(sessionData);
      const sessionBase64 = btoa(sessionJson);
      const THIRTY_DAYS = 60 * 60 * 24 * 30;

      cookies.set("tora_auth_token", sessionBase64, {
        path: "/",
        httpOnly: true,
        secure: !dev,
        sameSite: "strict",
        maxAge: THIRTY_DAYS,
      });

      redirect(303, "/dashboard");
    } catch (error: any) {
      console.error("Signup error:", error);
      return fail(400, {
        error: error.message || "Network error",
      });
    }
  },
};
