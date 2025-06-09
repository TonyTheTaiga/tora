import { json } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

export const DELETE: RequestHandler = async ({ params, locals }) => {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ message: "Authentication required" }, { status: 401 });
  }

  try {
    await locals.dbClient.deleteReference(params.slug, params.refId);
    return json({ message: "Reference deleted successfully" }, { status: 200 });
  } catch (error) {
    return json(
      {
        message:
          error instanceof Error
            ? error.message
            : "Unknown error deleting reference",
      },
      { status: 500 },
    );
  }
};
