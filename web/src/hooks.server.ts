import type { Handle } from "@sveltejs/kit";
import type { SessionData } from "$lib/types";

export const handle: Handle = async ({ event, resolve }) => {
  const auth_token = event.cookies.get("tora_auth_token");
  if (auth_token) {
    try {
      const sessionJson = atob(auth_token);
      const sessionData: SessionData = JSON.parse(sessionJson);
      const now = Math.floor(Date.now() / 1000);
      if (sessionData.expires_at > now) {
        event.locals.session = sessionData;
      } else {
        event.cookies.delete("tora_auth_token", { path: "/" });
      }
    } catch (error) {
      event.cookies.delete("tora_auth_token", { path: "/" });
    }
  }

  return resolve(event);
};
