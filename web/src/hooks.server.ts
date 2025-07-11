import type { Handle } from "@sveltejs/kit";
import type { SessionData } from "$lib/types";
import { ApiClient } from "$lib/api";

type RefreshResponse = {
  status: number;
  data: SessionData;
};

export const handle: Handle = async ({ event, resolve }) => {
  const auth_token = event.cookies.get("tora_auth_token");
  console.log("auth_token", auth_token);
  let apiClient: ApiClient;
  if (auth_token) {
    const sessionJson = atob(auth_token);
    const sessionData: SessionData = JSON.parse(sessionJson);
    const now = Math.floor(Date.now() / 1000);
    if (sessionData.expires_at > now) {
      event.locals.session = sessionData;
      apiClient = new ApiClient(undefined, sessionData.access_token);
    } else {
      apiClient = new ApiClient();
      const response = await apiClient.post<RefreshResponse>("/api/refresh", {
        refresh_token: sessionData.refresh_token,
      });
      if (response.status === 200) {
        const newSessionData: SessionData = {
          access_token: response.data.access_token,
          refresh_token: response.data.refresh_token,
          expires_in: response.data.expires_in,
          expires_at: response.data.expires_at,
          user: {
            id: response.data.user.id,
            email: response.data.user.email,
          },
        };
        event.locals.session = newSessionData;
        apiClient = new ApiClient(undefined, newSessionData.access_token);
      } else if (response.status === 401) {
        event.cookies.delete("tora_auth_token", { path: "/" });
      }
    }
  } else {
    apiClient = new ApiClient();
  }

  event.locals.apiClient = apiClient;
  return resolve(event);
};
