import type { Actions } from "./$types";
import type { PageServerLoad } from "./$types";
import { fail, redirect } from "@sveltejs/kit";
import type { Experiment, HyperParam } from "$lib/types";

const API_ROUTES = {
  GET_EXPERIMENTS: "/api/experiments",
  CREATE_EXPERIMENT: "/api/experiments",
  DELETE_EXPERIMENT: "/api/experiments/delete",
  UPDATE_EXPERIMENT: "/api/experiments/update",
  CREATE_REFERENCE: "/api/experiments/[slug]/ref",
} as const;

interface FormDataResult {
  hyperparams: HyperParam[];
  tags: string[];
  [key: string]: any;
}

export const load: PageServerLoad = async ({ fetch }) => {
  const response = await fetch(API_ROUTES.GET_EXPERIMENTS);
  const rawData = await response.json();
  const experiments = rawData.map(mapExperimentData);
  return { experiments };
};

export const actions: Actions = {
  create: async ({ request, fetch }) => handleCreate(request, fetch),
  delete: async ({ request, fetch }) => handleDelete(request, fetch),
  update: async ({ request, fetch }) => handleUpdate(request, fetch),
};

function mapExperimentData(exp: any): Experiment {
  return {
    id: exp.id,
    name: exp.name,
    description: exp.description,
    hyperparams: exp.hyperparams,
    createdAt: new Date(exp.createdAt),
    availableMetrics: exp.availableMetrics ?? undefined,
    tags: exp.tags,
  };
}

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
    hyperparams,
    tags,
  } = parseFormData(form);

  if (!name || !description) {
    return fail(400, { message: "Name and description are required" });
  }

  const response = await fetch(API_ROUTES.CREATE_EXPERIMENT, {
    method: "POST",
    body: JSON.stringify({ name, description, hyperparams, tags }),
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

  throw redirect(303, "/");
}

async function handleDelete(request: Request, fetch: Function) {
  const data = await request.formData();
  const id = data.get("id");

  if (!id) {
    return fail(400, { message: "ID is required" });
  }

  const response = await fetch(API_ROUTES.DELETE_EXPERIMENT, {
    method: "POST",
    body: JSON.stringify({ id }),
  });

  if (!response.ok) {
    return fail(500, { message: "Failed to delete experiment" });
  }

  throw redirect(303, "/");
}

async function handleUpdate(request: Request, fetch: Function) {
  const form = await request.formData();
  const {
    "experiment-id": id,
    "experiment-name": name,
    "experiment-description": description,
    "reference-id": referenceId,
    tags,
  } = parseFormData(form);

  if (!id || !name || !description) {
    return fail(400, { message: "ID, name, and description are required" });
  }

  const response = await fetch(`/api/experiments/${id}`, {
    method: "POST",
    body: JSON.stringify({ name, description, tags }),
  });

  if (!response.ok) {
    return fail(500, { message: "Failed to update experiment" });
  }

  if (referenceId) {
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

  return {
    success: true,
    message: "Experiment updated successfully!",
  };
}
