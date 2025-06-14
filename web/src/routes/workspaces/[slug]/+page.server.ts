import type { Actions, PageServerLoad } from "./$types";
import { startTimer, generateRequestId } from "$lib/utils/timing";
import type { Experiment, HyperParam } from "$lib/types";
import { fail, json, redirect } from "@sveltejs/kit";

interface FormDataResult {
  hyperparams: HyperParam[];
  tags: string[];
  [key: string]: any;
}

export const load: PageServerLoad = async ({ locals, params, parent }) => {
  const requestId = generateRequestId();
  const { workspaces } = await parent();

  const timer = startTimer("workspaces.load", {
    requestId,
  });
  const workspaceId = params.slug;
  try {
    let experiments: Experiment[] =
      await locals.dbClient.getExperiments(workspaceId);
    experiments = experiments.sort(
      (a, b) =>
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
    );

    const workspace = workspaces.find((w) => w.id === workspaceId);
    timer.end({ workspace_id: workspaceId });
    return { experiments, workspace };
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};

function parseFormData(formData: FormData): FormDataResult {
  const data = Object.fromEntries(formData);
  const result: FormDataResult = {
    hyperparams: [],
    tags: [],
  };
  const hyperparamMap = new Map<number, Partial<HyperParam>>();

  for (const [key, value] of Object.entries(data)) {
    if (typeof value !== "string") continue;

    if (key.startsWith("hyperparams.")) {
      const [, indexStr, field] = key.split(".");
      const index = Number(indexStr);
      const existing = hyperparamMap.get(index) ?? {};
      hyperparamMap.set(index, { ...existing, [field]: value });
    } else if (key.startsWith("tags.")) {
      const [, indexStr] = key.split(".");
      result.tags[Number(indexStr)] = value;
    } else {
      result[key] = value;
    }
  }

  result.hyperparams = [...hyperparamMap.values()].filter(
    (hp): hp is HyperParam => hp.key != null && hp.value != null,
  );
  result.tags = result.tags.filter(Boolean);

  return result;
}

export const actions: Actions = {
  create: async ({ request, locals }) => {
    if (!locals.user) {
      throw new Error("must be signed in to create a experiment");
    }

    const {
      "experiment-name": name,
      "experiment-description": description,
      "reference-id": referenceId,
      "workspace-id": workspaceId,
      tags,
      hyperparams,
      visibility,
    } = parseFormData(await request.formData());

    const experiment = await locals.dbClient.createExperiment(locals.user.id, {
      name,
      description,
      hyperparams,
      tags,
      visibility,
      workspaceId,
    });

    // Handle reference creation if provided
    if (referenceId && referenceId !== '') {
      try {
        await locals.dbClient.createReference(experiment.id, referenceId);
      } catch (error) {
        console.error("Failed to create experiment reference:", error);
      }
    }

    return { success: true };
  },
  update: async ({ request, locals }) => {
    const {
      "experiment-id": id,
      "experiment-name": name,
      "experiment-description": description,
      "reference-id": referenceId,
      tags,
      hyperparams,
      visibility,
    } = parseFormData(await request.formData());

    // Update experiment basic fields
    await locals.dbClient.updateExperiment(id, {
      name: name,
      description: description,
      tags: tags,
      visibility: visibility,
    });

    // Handle reference updates
    if (referenceId && referenceId !== '') {
      // Get current references to check if we need to update
      try {
        const currentReferences = await locals.dbClient.getReferenceChain(id);
        
        // If there's a current reference and it's different, delete it first
        if (currentReferences.length > 0 && currentReferences[0].id !== referenceId) {
          await locals.dbClient.deleteReference(id, currentReferences[0].id);
        }
        
        // If no current reference or it's different, create the new one
        if (currentReferences.length === 0 || currentReferences[0].id !== referenceId) {
          await locals.dbClient.createReference(id, referenceId);
        }
      } catch (error) {
        console.error("Failed to update experiment reference:", error);
      }
    } else {
      // If no reference provided, remove any existing references
      try {
        const currentReferences = await locals.dbClient.getReferenceChain(id);
        if (currentReferences.length > 0) {
          await locals.dbClient.deleteReference(id, currentReferences[0].id);
        }
      } catch (error) {
        console.error("Failed to remove experiment reference:", error);
      }
    }

    return { success: true };
  },

  delete: async ({ request, locals }) => {
    const data = await request.formData();
    const id = data.get("id");

    if (!id || typeof id !== "string") {
      return fail(400, {
        message: "A valid ID is required",
      });
    }
    await locals.dbClient.deleteExperiment(id);

    return { success: true };
  },
};
