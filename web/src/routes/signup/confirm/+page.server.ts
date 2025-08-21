import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ url, fetch }) => {
  const tokenHash = url.searchParams.get("token_hash");
  const confirmType = url.searchParams.get("confirm_type");

  if (!tokenHash || !confirmType) {
    return {
      success: false,
      message: "Malformed confirmation link!",
    };
  }

  try {
    const apiUrl = new URL("/api/signup/confirm", url.origin);
    apiUrl.searchParams.set("token_hash", tokenHash);
    apiUrl.searchParams.set("confirm_type", confirmType);

    const response = await fetch(apiUrl.toString());

    if (response.ok) {
      const result = await response.json();
      return {
        success: true,
        message: "Email confirmed successfully! Redirecting to dashboard...",
        data: result,
      };
    } else {
      return {
        success: false,
        message: "Confirmation failed. The link may be invalid or expired.",
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
