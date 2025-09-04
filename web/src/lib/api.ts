import { env } from "$env/dynamic/public";

const API_BASE_URL = env.PUBLIC_API_BASE_URL;

export class ApiClient {
  private baseUrl: string;
  private accessToken?: string;

  constructor(baseUrl?: string, accessToken?: string) {
    this.baseUrl = baseUrl || API_BASE_URL;
    this.accessToken = accessToken;
  }

  setAccessToken(token: string | null) {
    this.accessToken = token || undefined;
  }

  hasElevatedPermissions(): boolean {
    return !!this.accessToken;
  }

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers = new Headers(options.headers);
    headers.set("Content-Type", "application/json");

    if (this.accessToken) {
      headers.set("Authorization", `Bearer ${this.accessToken}`);
    }

    const defaultOptions: RequestInit = {
      credentials: "include",
      headers,
    };

    const startedAt = Date.now();
    const method = (options.method || "GET").toString().toUpperCase();
    let response: Response;
    try {
      response = await fetch(url, { ...defaultOptions, ...options });
    } catch (err) {
      const elapsedMs = Date.now() - startedAt;
      try {
        console.error(
          `[api] ${method} ${endpoint} -> error in ${elapsedMs}ms: ${err instanceof Error ? err.message : String(err)}`,
        );
      } catch (_) {
        // best-effort logging
      }
      throw err;
    }

    const elapsedMs = Date.now() - startedAt;
    try {
      // Log a concise line for latency monitoring
      // Example: [api] GET /workspaces -> 200 in 123ms
      console.info(
        `[api] ${method} ${endpoint} -> ${response.status} in ${elapsedMs}ms`,
      );
    } catch (_) {
      // best-effort logging; never fail the request because of logging
    }

    if (!response.ok) {
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}`,
      );
    }
    const contentType = response.headers.get("content-type");
    if (
      contentType &&
      contentType.includes("application/json") &&
      response.status !== 204
    ) {
      return response.json();
    }

    return response.text() as T;
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: "GET" });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: "POST",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: "PUT",
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: "DELETE" });
  }
}
