import { fail } from "@sveltejs/kit";
import type { Actions } from "./$types";

export const actions: Actions = {
  default: async ({ request, locals }) => {
    const data = await request.formData();
    const email = data.get("email") as string;
    const password = data.get("password") as string;

    if (!email || !password) {
      return fail(400, {
        error: "Email and password are required",
      });
    }

    try {
      await locals.apiClient.post("/api/signup", {
        email,
        password,
      });

      return { success: true };
    } catch (error: any) {
      console.error("Signup error:", error);
      return fail(400, {
        error: error.message || "Network error",
      });
    }
  },
};