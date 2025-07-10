import { env } from "$env/dynamic/public";
import type { SessionData } from "$lib/types";

const API_BASE_URL = env.PUBLIC_API_BASE_URL || "http://localhost:8080";

export class ServerApiClient {
  private baseUrl: string;
  private accessToken?: string;

  constructor(baseUrl?: string, accessToken?: string) {
    this.baseUrl = baseUrl || API_BASE_URL;
    this.accessToken = accessToken;
  }

  async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...((options.headers as Record<string, string>) || {}),
    };

    // Add Authorization header if we have an access token
    if (this.accessToken) {
      headers.Authorization = `Bearer ${this.accessToken}`;
    }

    const requestOptions: RequestInit = {
      ...options,
      headers,
    };

    const response = await fetch(url, requestOptions);

    if (!response.ok) {
      throw new Error(
        `API request failed: ${response.status} ${response.statusText}`,
      );
    }

    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
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

// Helper function to create an authenticated API client from session data
export function createAuthenticatedApiClient(
  sessionData?: SessionData | null,
): ServerApiClient {
  return new ServerApiClient(API_BASE_URL, sessionData?.access_token);
}

// Default client for non-authenticated requests
export const serverApiClient = new ServerApiClient();
