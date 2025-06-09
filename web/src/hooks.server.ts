import { createServerClient } from "@supabase/ssr";
import { type Handle, redirect } from "@sveltejs/kit";
import { sequence } from "@sveltejs/kit/hooks";
import { createDbClient } from "$lib/server/database";
import { createHash } from "crypto";

import {
  PUBLIC_SUPABASE_URL,
  PUBLIC_SUPABASE_ANON_KEY,
} from "$env/static/public";

const supabase: Handle = async ({ event, resolve }) => {
  /**
   * Creates a Supabase client specific to this server request.
   *
   * The Supabase client gets the Auth token from the request cookies.
   */
  event.locals.supabase = createServerClient(
    PUBLIC_SUPABASE_URL,
    PUBLIC_SUPABASE_ANON_KEY,
    {
      cookies: {
        getAll: () => event.cookies.getAll(),
        /**
         * SvelteKit's cookies API requires `path` to be explicitly set in
         * the cookie options. Setting `path` to `/` replicates previous/
         * standard behavior.
         */
        setAll: (cookiesToSet) => {
          cookiesToSet.forEach(({ name, value, options }) => {
            event.cookies.set(name, value, { ...options, path: "/" });
          });
        },
      },
    },
  );

  event.locals.dbClient = createDbClient(event.locals.supabase);

  /**
   * Unlike `supabase.auth.getSession()`, which returns the session _without_
   * validating the JWT, this function also calls `getUser()` to validate the
   * JWT before returning the session.
   */
  event.locals.safeGetSession = async () => {
    const {
      data: { session },
    } = await event.locals.supabase.auth.getSession();
    if (!session) {
      return { session: null, user: null };
    }

    const {
      data: { user },
      error,
    } = await event.locals.supabase.auth.getUser();
    if (error) {
      // JWT validation has failed
      return { session: null, user: null };
    }

    return { session, user };
  };

  return resolve(event, {
    filterSerializedResponseHeaders(name) {
      /**
       * Supabase libraries use the `content-range` and `x-supabase-api-version`
       * headers, so we need to tell SvelteKit to pass it through.
       */
      return name === "content-range" || name === "x-supabase-api-version";
    },
  });
};

const finalize: Handle = async ({ event, resolve }) => {
  const { session, user } = await event.locals.safeGetSession();
  event.locals.session = session;
  event.locals.user = user;

  if (
    event.url.pathname.startsWith("/api") &&
    !event.locals.user &&
    event.request.headers.get("x-api-key")
  ) {
    const apiKey = event.request.headers.get("x-api-key");

    try {
      if (apiKey) {
        const keyHash = createHash("sha256").update(apiKey).digest("hex");

        const { data: keyData, error: keyError } = await event.locals.supabase
          .from("api_keys")
          .select("user_id")
          .eq("key_hash", keyHash)
          .eq("revoked", false)
          .single();

        if (keyError) {
          console.error("Supabase API key query error:", keyError.message);
        }

        if (keyData && keyData.user_id) {
          event.locals.user = {
            id: keyData.user_id,
          };

          try {
            await event.locals.supabase
              .from("api_keys")
              .update({ last_used: new Date().toISOString() })
              .eq("key_hash", keyHash)
              .eq("revoked", false);
          } catch (updateErr) {
            console.warn(
              `Failed to update API key last_used for user_id: ${keyData.user_id}:`,
              updateErr instanceof Error ? updateErr.message : updateErr,
            );
          }
        } else {
          console.warn(
            "API key provided, but no valid user found or key is revoked.",
          );
        }
      } else {
        console.warn("API key header was present but empty.");
      }
    } catch (err) {
      console.error(
        "Unexpected error during API key validation:",
        err instanceof Error ? err.message : err,
        err instanceof Error ? err.stack : "",
      );
    }
  }

  return resolve(event);
};

// const authGuard: Handle = async ({ event, resolve }) => {
//   if (event.url.pathname.startsWith("/api")) {
//     return resolve(event);
//   }

//   const { session, user } = await event.locals.safeGetSession();
//   event.locals.session = session;
//   event.locals.user = user;

//   if (!event.locals.session && event.url.pathname.startsWith("/private")) {
//     redirect(303, "/auth");
//   }

//   if (event.locals.session && event.url.pathname === "/auth") {
//     redirect(303, "/private");
//   }
//   return resolve(event);
// };

export const handle: Handle = sequence(supabase, finalize);
