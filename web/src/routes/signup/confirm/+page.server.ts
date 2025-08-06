import type { PageServerLoad } from "./$types";
import { error } from "@sveltejs/kit";

export const load: PageServerLoad = async ({ url, fetch }) => {
  // Extract all possible parameters that Supabase might need
  const tokenHash = url.searchParams.get("token_hash");
  const confirmType =
    url.searchParams.get("confirm_type") || url.searchParams.get("type");
  const redirectTo = url.searchParams.get("redirect_to");

  if (!tokenHash) {
    return {
      success: false,
      message: "Missing confirmation token. Please check your email link.",
    };
  }

  try {
    const apiUrl = new URL("/api/signup/confirm", url.origin);
    apiUrl.searchParams.set("token_hash", tokenHash);
    if (confirmType) {
      apiUrl.searchParams.set("confirm_type", confirmType);
    }
    if (redirectTo) {
      apiUrl.searchParams.set("redirect_to", redirectTo);
    }
    for (const [key, value] of url.searchParams) {
      if (!apiUrl.searchParams.has(key)) {
        apiUrl.searchParams.set(key, value);
      }
    }
    const response = await fetch(apiUrl.toString());

    if (response.ok) {
      const result = await response.json();
      return {
        success: true,
        message: "Email confirmed successfully! Redirecting to dashboard...",
        data: result,
      };
    } else {
      // Handle API errors
      const errorData = await response.json().catch(() => ({}));
      return {
        success: false,
        message:
          errorData.message ||
          "Confirmation failed. The link may be invalid or expired.",
      };
    }
  } catch (err) {
    console.error("Signup confirmation error:", err);
    return {
      success: false,
      message:
        "An error occurred during confirmation. Please try again or contact support.",
    };
  }
};
