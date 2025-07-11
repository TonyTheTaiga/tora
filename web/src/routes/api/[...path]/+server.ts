import { env } from "$env/dynamic/public";
import { error } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";

const RUST_API_BASE_URL = env.PUBLIC_API_BASE_URL;

export const GET: RequestHandler = async ({ request, params, fetch }) => {
  return proxyRequest(request, params.path, fetch);
};

export const POST: RequestHandler = async ({ request, params, fetch }) => {
  return proxyRequest(request, params.path, fetch);
};

export const PUT: RequestHandler = async ({ request, params, fetch }) => {
  return proxyRequest(request, params.path, fetch);
};

export const DELETE: RequestHandler = async ({ request, params, fetch }) => {
  return proxyRequest(request, params.path, fetch);
};

export const PATCH: RequestHandler = async ({ request, params, fetch }) => {
  return proxyRequest(request, params.path, fetch);
};

async function proxyRequest(
  request: Request,
  path: string,
  fetch: typeof globalThis.fetch,
) {
  const url = new URL(request.url);
  const rustBackendUrl = `${RUST_API_BASE_URL}/api/${path}${url.search}`;
  const headers = new Headers(request.headers);
  // Remove host header as it will be set automatically by the fetch to the backend
  headers.delete("host");

  const options: RequestInit = {
    method: request.method,
    headers: headers,
    // For GET/HEAD requests, body should be null
    body:
      request.method === "GET" || request.method === "HEAD"
        ? null
        : await request.arrayBuffer(),
  };

  try {
    const response = await fetch(rustBackendUrl, options);

    const proxyResponse = new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });

    return proxyResponse;
  } catch (e) {
    console.error(`Proxy error for ${rustBackendUrl}:`, e);
    error(
      500,
      `Failed to proxy request to backend: ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}
