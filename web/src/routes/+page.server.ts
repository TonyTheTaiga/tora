import { fail } from "@sveltejs/kit";
import { generateRequestId, startTimer } from "$lib/utils/timing";
import type { Actions, PageServerLoad } from "./$types";
import type { HyperParam } from "$lib/types";

const API = {
  getExperiments: (origin: string, workspaceId?: string) => {
    const url = new URL("/api/experiments", origin);
    if (workspaceId) {
      url.searchParams.set("workspace", workspaceId);
    }
    return url;
  },
  createExperiment: "/api/experiments",
  deleteExperiment: (id: string) => `/api/experiments/${id}`,
  updateExperiment: (id: string) => `/api/experiments/${id}`,
  createReference: (slug: string) => `/api/experiments/${slug}/ref`,
  getReferences: (id: string) => `/api/experiments/${id}/ref`,
  deleteReference: (id: string, refId: string) =>
    `/api/experiments/${id}/ref/${refId}`,
} as const;

interface FormDataResult {
  hyperparams: HyperParam[];
  tags: string[];
  [key: string]: any;
}

export const load: PageServerLoad = async ({ fetch, locals, parent, url }) => {
  const requestId = generateRequestId();
  const timer = startTimer("home.load", {
    requestId,
  });
  try {
    timer.end({});
    return {};
  } catch (err) {
    timer.end({
      error: err instanceof Error ? err.message : "Unknown error",
    });
    throw err;
  }
};

// --- Form Actions ---
export const actions: Actions = {
  create: async ({ request, fetch }) => handleCreate(request, fetch),
  delete: async ({ request, fetch }) => handleDelete(request, fetch),
  update: async ({ request, fetch }) => handleUpdate(request, fetch),
};

async function handleCreate(request: Request, fetch: typeof window.fetch) {
  const form = await request.formData();
  const data = parseFormData(form);

  const {
    "experiment-name": name,
    "experiment-description": description,
    "reference-id": referenceId,
  } = data;

  if (!name || !description) {
    return fail(400, {
      message: "Name and description are required",
    });
  }

  const response = await fetch(API.createExperiment, {
    method: "POST",
    body: JSON.stringify({
      ...data,
      name,
      description,
    }),
  });

  if (!response.ok) {
    return fail(500, {
      message: "Failed to create experiment",
    });
  }

  if (referenceId) {
    const { experiment } = await response.json();
    const refResponse = await fetch(API.createReference(experiment.id), {
      method: "POST",
      body: JSON.stringify({
        referenceId,
      }),
    });

    if (!refResponse.ok) {
      return fail(500, {
        message: "Failed to create reference",
      });
    }
  }

  return {
    success: true,
  };
}

async function handleDelete(request: Request, fetch: typeof window.fetch) {
  const data = await request.formData();
  const id = data.get("id");

  if (!id || typeof id !== "string") {
    return fail(400, {
      message: "A valid ID is required",
    });
  }

  const response = await fetch(API.deleteExperiment(id), {
    method: "DELETE",
  });
  if (!response.ok) {
    return fail(500, {
      message: "Failed to delete experiment",
    });
  }

  return {
    success: true,
  };
}

async function handleUpdate(request: Request, fetch: typeof window.fetch) {
  const form = await request.formData();
  const data = parseFormData(form);

  const {
    "experiment-id": id,
    "experiment-name": name,
    "experiment-description": description,
    "reference-id": referenceId,
  } = data;

  if (!id || !name || !description) {
    return fail(400, {
      message: "ID, name, and description are required",
    });
  }

  const response = await fetch(API.updateExperiment(id), {
    method: "POST",
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    return fail(500, {
      message: "Failed to update experiment",
    });
  }

  try {
    await replaceExperimentReference(id, referenceId, fetch);
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Could not update references";
    return fail(500, {
      message,
    });
  }

  return {
    success: true,
  };
}

// --- Helper Functions ---
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

async function replaceExperimentReference(
  id: string,
  newReferenceId: any,
  fetch: typeof window.fetch,
) {
  const res = await fetch(API.getReferences(id));
  if (!res.ok) throw new Error("Failed to fetch existing references.");
  const currentRefIds: string[] = await res.json();

  await Promise.all(
    currentRefIds.map((refId) =>
      fetch(API.deleteReference(id, refId), {
        method: "DELETE",
      }),
    ),
  );

  if (newReferenceId && typeof newReferenceId === "string") {
    const addRes = await fetch(API.createReference(id), {
      method: "POST",
      body: JSON.stringify({
        referenceId: newReferenceId,
      }),
    });
    if (!addRes.ok) throw new Error("Failed to create new reference.");
  }
}
