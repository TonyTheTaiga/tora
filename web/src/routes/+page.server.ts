import type { Actions } from "./$types";
import type { PageServerLoad } from "./$types";
import { fail } from "@sveltejs/kit";
import type { HyperParam, Experiment, Metric } from "$lib/types";

const API_ROUTES = {
  GET_EXPERIMENTS: "/api/experiments",
  CREATE_EXPERIMENT: "/api/experiments",
  UPDATE_EXPERIMENT: "/api/experiments/update",
  CREATE_REFERENCE: "/api/experiments/[slug]/ref",
} as const;

interface FormDataResult {
  hyperparams: HyperParam[];
  tags: string[];
  [key: string]: any;
}

export const load: PageServerLoad = async ({
  fetch,
  locals,
  parent,
  url,
  depends,
}) => {
  const { session } = await locals.safeGetSession();
  const { currentWorkspace } = await parent();

  const apiUrl = new URL(API_ROUTES.GET_EXPERIMENTS, url.origin);
  if (currentWorkspace) {
    apiUrl.searchParams.set("workspace", currentWorkspace.id);
  }

  const response = await fetch(apiUrl.toString());
  if (!response.ok) {
    console.error(
      `Failed to fetch experiments: ${response.status} ${response.statusText}`,
    );
    return { experiments: [], session, error: "Failed to load experiments." };
  }

  let experiments: Experiment[] = await response.json();
  return { experiments, session };
};

export const actions: Actions = {
  create: async ({ request, fetch }) => handleCreate(request, fetch),
  delete: async ({ request, fetch }) => handleDelete(request, fetch),
  update: async ({ request, fetch }) => handleUpdate(request, fetch),
  switchWorkspace: async ({ request, cookies }) => {
    const formData = await request.formData();
    const workspaceId = formData.get("workspaceId");

    if (!workspaceId || typeof workspaceId !== "string") {
      return fail(400, { message: "Workspace ID is required" });
    }

    cookies.set("current_workspace", workspaceId, {
      path: "/",
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    return { success: true };
  },
};

function parseFormData(formData: FormData): FormDataResult {
  const obj = Object.fromEntries(formData);
  const result: FormDataResult = {
    hyperparams: [],
    tags: [],
  };

  Object.entries(obj).forEach(([key, value]) => {
    if (key.startsWith("hyperparams.")) {
      const [_, index, field] = key.split(".");
      const idx = Number(index);

      if (!result.hyperparams[idx]) {
        result.hyperparams[idx] = { key: value as string, value: "" };
      } else {
        result.hyperparams[idx].value = value as string | number;
      }
    } else if (key.startsWith("tags.")) {
      const [_, index] = key.split(".");
      const idx = Number(index);

      if (!result.tags[idx]) {
        result.tags[idx] = value as string;
      }
    } else {
      result[key] = value;
    }
  });

  return {
    ...result,
    hyperparams: result.hyperparams.filter(Boolean),
    tags: result.tags.filter(Boolean),
  };
}

async function handleCreate(request: Request, fetch: Function) {
  const form = await request.formData();
  const {
    "experiment-name": name,
    "experiment-description": description,
    "reference-id": referenceId,
    visibility,
    hyperparams,
    tags,
  } = parseFormData(form);

  if (!name || !description) {
    return fail(400, { message: "Name and description are required" });
  }

  const response = await fetch(API_ROUTES.CREATE_EXPERIMENT, {
    method: "POST",
    body: JSON.stringify({ name, description, hyperparams, tags, visibility }),
  });

  if (!response.ok) {
    return fail(500, { message: "Failed to create experiment" });
  }

  if (referenceId) {
    const {
      experiment: { id },
    } = await response.json();
    const referenceResponse = await fetch(
      API_ROUTES.CREATE_REFERENCE.replace("[slug]", id),
      {
        method: "POST",
        body: JSON.stringify({ referenceId }),
      },
    );

    if (!referenceResponse.ok) {
      return fail(500, { message: "Failed to create reference" });
    }
  }

  return { success: true };
}

async function handleDelete(request: Request, fetch: Function) {
  const data = await request.formData();
  const id = data.get("id");

  if (!id) {
    return fail(400, { message: "ID is required" });
  }

  const response = await fetch(`/api/experiments/${id}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    return fail(500, { message: "Failed to delete experiment" });
  }

  return { success: true };
}

async function handleUpdate(request: Request, fetch: Function) {
  const form = await request.formData();
  const {
    "experiment-id": id,
    "experiment-name": name,
    "experiment-description": description,
    "reference-id": referenceId,
    visibility,
    tags,
  } = parseFormData(form);

  if (!id || !name || !description) {
    return fail(400, { message: "ID, name, and description are required" });
  }

  const response = await fetch(`/api/experiments/${id}`, {
    method: "POST",
    body: JSON.stringify({ name, description, visibility, tags }),
  });

  if (!response.ok) {
    return fail(500, { message: "Failed to update experiment" });
  }

  const refResponse = await fetch(`/api/experiments/${id}/ref`);
  const currentRefs = await refResponse.json();

  if (referenceId) {
    if (currentRefs.length > 0) {
      for (const refId of currentRefs) {
        if (refId !== id) {
          await fetch(`/api/experiments/${id}/ref/${refId}`, {
            method: "DELETE",
          });
        }
      }
    }

    const referenceResponse = await fetch(
      API_ROUTES.CREATE_REFERENCE.replace("[slug]", id),
      {
        method: "POST",
        body: JSON.stringify({ referenceId }),
      },
    );

    if (!referenceResponse.ok) {
      return fail(500, { message: "Failed to create reference" });
    }
  } else if (currentRefs.length > 0) {
    for (const refId of currentRefs) {
      if (refId !== id) {
        await fetch(`/api/experiments/${id}/ref/${refId}`, {
          method: "DELETE",
        });
      }
    }
  }

  return {
    success: true,
  };
}
