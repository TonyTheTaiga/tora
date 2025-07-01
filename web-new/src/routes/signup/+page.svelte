<script lang="ts">
  import { goto } from "$app/navigation";
  import { User, Mail, Lock, LogIn, Loader2 } from "@lucide/svelte";
  import { enhance } from "$app/forms";

  let submitting = $state(false);
</script>

<div
  class="flex items-center justify-center min-h-[calc(100vh-2rem)] font-mono"
>
  <div class="w-full max-w-md">
    <div
      class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 overflow-hidden"
    >
      <!-- Header -->
      <div
        class="px-6 py-4 border-b border-ctp-surface0 flex items-center gap-2"
      >
        <User size={20} class="text-ctp-mauve" />
        <h2 class="text-xl text-ctp-text">create account</h2>
      </div>

      <!-- Form -->
      <form
        method="POST"
        autocomplete="on"
        action=""
        name="login-form"
        class="p-6 space-y-5"
        use:enhance={() => {
          submitting = true;
          return async ({ result }) => {
            submitting = false;
            if (result.type === "redirect") {
              goto(result.location);
            }
          };
        }}
      >
        <!-- Email field -->
        <div class="space-y-2">
          <label class="text-sm text-ctp-subtext0 block" for="email">
            email address
          </label>
          <div class="relative">
            <div
              class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"
            >
              <Mail size={18} class="text-ctp-subtext0" />
            </div>
            <input
              id="email"
              name="email"
              type="email"
              autocomplete="email"
              required
              class="w-full pl-10 pr-4 py-3 bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-text focus:outline-none focus:ring-1 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 font-mono"
              placeholder="your.email@example.com"
            />
          </div>
        </div>

        <!-- Password field -->
        <div class="space-y-2">
          <label class="text-sm text-ctp-subtext0 block" for="password">
            password
          </label>
          <div class="relative">
            <div
              class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"
            >
              <Lock size={18} class="text-ctp-subtext0" />
            </div>
            <input
              id="password"
              name="password"
              type="password"
              minlength="6"
              autocomplete="new-password"
              required
              class="w-full pl-10 pr-4 py-3 bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-text focus:outline-none focus:ring-1 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 font-mono"
              placeholder="create a strong password"
            />
          </div>
        </div>

        <!-- Button actions -->
        <div class="pt-2">
          <button
            type="submit"
            class="w-full flex items-center justify-center gap-2 bg-ctp-blue/20 border border-ctp-blue/40 py-3 px-4 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all font-mono disabled:opacity-50"
            disabled={submitting}
          >
            {#if submitting}
              <Loader2 size={18} class="animate-spin" />
              creating...
            {:else}
              <LogIn size={18} />
              create account
            {/if}
          </button>
        </div>
      </form>

      <!-- Footer -->
      <div
        class="flex justify-end gap-3 pt-2 pb-6 px-6 border-t border-ctp-surface0"
      >
        <p class="text-sm text-ctp-subtext0 pt-3">already have an account?</p>
        <button
          type="button"
          class="inline-flex items-center justify-center px-5 py-2.5 bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors font-mono"
          onclick={() => {
            goto("/login");
          }}
        >
          log in
        </button>
      </div>
    </div>
  </div>
</div>
