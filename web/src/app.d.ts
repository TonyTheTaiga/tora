// See https://svelte.dev/docs/kit/types#app.d.ts

import type { Database } from "$lib/server/database.types";
import type { Session, SupabaseClient, User } from "@supabase/supabase-js";
import type { createDbClient } from "$lib/server/database";

// for information about these interfaces
declare global {
  namespace App {
    // interface Error {}
    interface Locals {
      supabase: SupabaseClient<Database>;
      adminSupabaseClient: SupabaseClient<Database>;
      dbClient: ReturnType<typeof createDbClient>;
      safeGetSession: () => Promise<{
        session: Session | null;
        user: User | null;
      }>;
      session: Session | null;
      user: User | { id: string } | null;
    }
    interface PageData {
      session: Session | null;
    }
    // interface PageState {}
    // interface Platform {}
  }
}

export {};
