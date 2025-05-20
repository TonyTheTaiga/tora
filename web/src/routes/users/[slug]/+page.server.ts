import { error } from "@sveltejs/kit";
import type { PageServerLoad, Actions } from "./$types";

export const load: PageServerLoad = async ({ params, locals, fetch }) => {
  const slug = params.slug;

  // Make sure the user is authenticated and only allow viewing of own profile
  if (!locals.user) {
    throw error(401, "Must be logged in to view profile");
  }

  // Only allow users to view their own profile
  if (locals.user.id !== slug) {
    throw error(403, "You can only view your own profile");
  }

  try {
    let apiKeys = [];
    try {
      const response = await fetch("/api/keys");
      if (response.ok) {
        const data = await response.json();
        apiKeys = data.keys;
      }
    } catch (err) {
      console.error("Error fetching API keys:", err);
    }

    return {
      slug,
      user: {
        id: locals.user.id,
        email: 'email' in locals.user ? locals.user.email : '',
        name: 'user_metadata' in locals.user && locals.user.user_metadata?.full_name || "User",
        username:
          ('user_metadata' in locals.user && locals.user.user_metadata?.username) ||
          ('email' in locals.user && locals.user.email?.split("@")[0]) ||
          "user",
        apiKeys,
      },
    };
  } catch (err) {
    console.error("Error loading user profile:", err);
    throw error(500, "Error loading profile");
  }
};

export const actions: Actions = {
  createApiKey: async ({ request, locals, fetch }) => {
    if (!locals.user) {
      throw error(401, "Unauthorized");
    }

    const formData = await request.formData();
    const name = formData.get("name")?.toString();

    if (!name) {
      return { success: false, error: "Key name is required" };
    }

    try {
      const response = await fetch("/api/keys", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name }),
      });

      if (!response.ok) {
        const errorData: { message?: string } = await response.json();
        return {
          success: false,
          error: errorData.message || "Failed to create API key",
        };
      }

      const data = await response.json();
      return {
        success: true,
        key: data.key,
      };
    } catch (err) {
      console.error("Error creating API key:", err);
      return {
        success: false,
        error: err instanceof Error ? err.message : "Server error",
      };
    }
  },

  deleteApiKey: async ({ request, locals, fetch }) => {
    if (!locals.user) {
      throw error(401, "Unauthorized");
    }

    const formData = await request.formData();
    const keyId = formData.get("id")?.toString();

    if (!keyId) {
      return { success: false, error: "Key ID is required" };
    }

    try {
      const response = await fetch(`/api/keys?id=${keyId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        const errorData: { message?: string } = await response.json();
        return {
          success: false,
          error: errorData.message || "Failed to delete API key",
        };
      }

      return { success: true };
    } catch (err) {
      console.error("Error deleting API key:", err);
      return {
        success: false,
        error: err instanceof Error ? err.message : "Server error",
      };
    }
  },
};
