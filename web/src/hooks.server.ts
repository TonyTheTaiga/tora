import type { Handle } from "@sveltejs/kit";
import type { SessionData } from "$lib/types";
import { ApiClient } from "$lib/api";

export const handle: Handle = async ({ event, resolve }) => {
  const auth_token = event.cookies.get("tora_auth_token");
  
  // Always create an API client (authenticated or not)
  let apiClient: ApiClient;
  
  if (auth_token) {
    try {
      const sessionJson = atob(auth_token);
      const sessionData: SessionData = JSON.parse(sessionJson);
      const now = Math.floor(Date.now() / 1000);
      if (sessionData.expires_at > now) {
        event.locals.session = sessionData;
        // Create authenticated API client
        apiClient = new ApiClient(undefined, sessionData.access_token);
      } else {
        event.cookies.delete("tora_auth_token", { path: "/" });
        // Create unauthenticated API client
        apiClient = new ApiClient();
      }
    } catch (error) {
      event.cookies.delete("tora_auth_token", { path: "/" });
      // Create unauthenticated API client
      apiClient = new ApiClient();
    }
  } else {
    // Create unauthenticated API client
    apiClient = new ApiClient();
  }
  
  event.locals.apiClient = apiClient;
  return resolve(event);
};
