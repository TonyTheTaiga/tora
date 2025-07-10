// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
import type { SessionData } from "$lib/types";
import type { ApiClient } from "$lib/api";

declare global {
  namespace App {
    // interface Error {}
    interface Locals {
      session: SessionData | null;
      apiClient: ApiClient;
    }
    // interface PageData {}
    // interface PageState {}
  }
}

export {};
