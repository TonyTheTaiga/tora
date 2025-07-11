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
    const response = await locals.apiClient.post<LoginResponse>("/api/login", {
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
    cookies.set("tora_auth_token", sessionBase64, {
      path: "/",
      httpOnly: true,
      secure: !dev,
      sameSite: "strict",
      maxAge: response.data.expires_in,
    });

    redirect(303, "/workspaces");
  },
};
