import type { PageServerLoad } from "./$types";

export const load: PageServerLoad = async ({ url, locals }) => {
  const tokenHash = url.searchParams.get("token_hash");
  const confirmType = url.searchParams.get("confirm_type");

  if (!tokenHash || !confirmType) {
    return {
      success: false,
      message: "Malformed confirmation link!",
    };
  }

  try {
    const result = await locals.apiClient.get<any>(
      `/signup/confirm?token_hash=${encodeURIComponent(tokenHash)}&confirm_type=${encodeURIComponent(confirmType)}`,
    );
    return {
      success: true,
      message: "Email confirmed successfully! Redirecting to dashboard...",
      data: result,
    };
  } catch (err) {
    console.error("Signup confirmation error:", err);
    return {
      success: false,
      message:
        "An error occurred during confirmation. Please try again or contact support.",
    };
  }
};
