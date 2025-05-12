import { json, error } from "@sveltejs/kit";
import { createReference, getReferenceChain, getExperiment } from "$lib/server/database";

export async function GET({
  params: { slug },
}: {
  params: { slug: string },
  locals: { user: { id: string } | null }
}) {
  const references = await getReferenceChain(slug);
  return json(references.map((experiment) => experiment.id));

}

export async function POST({
  params: { slug },
  request,
  locals,
}: {
  params: { slug: string };
  request: Request;
  locals: { user: { id: string } | null };
}) {
  const userId = locals.user?.id;
  if (!userId) {
    return json({ message: "Authentication required" }, { status: 401 });
  }

  try {
    const { referenceId } = await request.json();
    try {
      await getExperiment(slug, userId);
      await getExperiment(referenceId, userId);
    } catch (accessError) {
      return json({ message: "Access denied to one or both experiments" }, { status: 403 });
    }

    await createReference(slug, referenceId);
    return json({ message: "Reference created successfully" }, { status: 201 });
  } catch (error) {
    return json({
      message: error instanceof Error ? error.message : "Unknown error creating reference"
    }, { status: 500 });
  }
}
