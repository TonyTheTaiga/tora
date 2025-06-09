import { json } from "@sveltejs/kit";
import { deleteReference } from "$lib/server/database";

export async function DELETE({
  params: { slug, refId },
  locals,
}: {
  params: { slug: string; refId: string };
  locals: { user: { id: string } | null };
}) {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ message: "Authentication required" }, { status: 401 });
  }

  try {
    await deleteReference(slug, refId);
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
}
