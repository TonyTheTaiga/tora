import { env } from "$env/dynamic/public";
import { error, type Cookies } from "@sveltejs/kit";
import type { RequestHandler } from "./$types";
import type { SessionData } from "$lib/types";

const RUST_API_BASE_URL = env.PUBLIC_API_BASE_URL;

export const GET: RequestHandler = async ({
  request,
  params,
  fetch,
  cookies,
}) => {
  return proxyRequest(request, params.proxyPath, fetch, cookies);
};

export const POST: RequestHandler = async ({
  request,
  params,
  fetch,
  cookies,
}) => {
  return proxyRequest(request, params.proxyPath, fetch, cookies);
};

export const PUT: RequestHandler = async ({
  request,
  params,
  fetch,
  cookies,
}) => {
  return proxyRequest(request, params.proxyPath, fetch, cookies);
};

export const DELETE: RequestHandler = async ({
  request,
  params,
  fetch,
  cookies,
}) => {
  return proxyRequest(request, params.proxyPath, fetch, cookies);
};

export const PATCH: RequestHandler = async ({
  request,
  params,
  fetch,
  cookies,
}) => {
  return proxyRequest(request, params.proxyPath, fetch, cookies);
};

async function proxyRequest(
  request: Request,
  path: string,
  fetch: typeof globalThis.fetch,
  cookies: Cookies,
) {
  const url = new URL(request.url);
  const rustBackendUrl = `${RUST_API_BASE_URL}/${path}${url.search}`;

  const headers = new Headers(request.headers);
  headers.delete("host");
  headers.delete("cookie");

  const auth_token = cookies.get("tora_auth_token");
  if (auth_token) {
    const sessionJson = atob(auth_token);
    const sessionData: SessionData = JSON.parse(sessionJson);
    headers.set("Authorization", `Bearer ${sessionData.access_token}`);
  }

  let body: BodyInit | null = null;

  if (request.method !== "GET" && request.method !== "HEAD") {
    const contentType = request.headers.get("content-type");

    if (contentType?.includes("multipart/form-data")) {
      body = await request.formData();
      headers.delete("content-type");
    } else if (contentType?.includes("application/x-www-form-urlencoded")) {
      body = await request.text();
    } else if (contentType?.includes("application/json")) {
      body = await request.text();
    } else {
      body = await request.arrayBuffer();
    }
  }

  const options: RequestInit = {
    method: request.method,
    headers: headers,
    body: body,
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
