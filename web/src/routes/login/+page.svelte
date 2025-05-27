<script lang="ts">
  import { LogIn, User, Lock, Mail } from "lucide-svelte";
  import { goto } from "$app/navigation";
</script>

<div class="flex items-center justify-center h-full">
  <div class="w-full max-w-md">
    <div
      class="bg-ctp-mantle shadow-lg rounded-xl overflow-hidden border border-ctp-surface0"
    >
      <!-- Header -->
      <div
        class="px-6 py-4 border-b border-ctp-surface0 flex items-center gap-2"
      >
        <User size={20} class="text-ctp-mauve" />
        <h2 class="text-xl font-medium text-ctp-text">Sign In</h2>
      </div>

      <!-- Form -->
      <form
        method="POST"
        autocomplete="on"
        action=""
        name="login-form"
        class="p-6 space-y-5"
      >
        <!-- Email field -->
        <div class="space-y-2">
          <label
            class="text-sm font-medium text-ctp-subtext0 block"
            for="email"
          >
            Email Address
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
              class="w-full pl-10 pr-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-blue transition-all placeholder-ctp-overlay0 shadow-sm"
              placeholder="your.email@example.com"
            />
          </div>
        </div>

        <!-- Password field -->
        <div class="space-y-2">
          <label
            class="text-sm font-medium text-ctp-subtext0 block"
            for="password"
          >
            Password
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
              autocomplete="current-password"
              required
              class="w-full pl-10 pr-4 py-3 bg-ctp-base border-0 rounded-lg text-ctp-text focus:outline-none focus:ring-2 focus:ring-ctp-mauve transition-all placeholder-ctp-overlay0 shadow-sm"
              placeholder="Enter your password"
            />
          </div>
        </div>

        <!-- Button actions -->
        <div class="pt-2">
          <button
            type="submit"
            class="w-full flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve py-3 px-4 text-ctp-crust font-medium hover:shadow-lg transition-all"
          >
            <LogIn size={18} />
            Sign In
          </button>
        </div>
      </form>

      <!-- Footer -->
      <div
        class="flex justify-end gap-3 pt-2 pb-6 px-6 border-t border-ctp-surface0"
      >
        <p class="text-sm text-ctp-subtext0 pt-3">Don't Have an Account?</p>
        <button
          type="button"
          class="inline-flex items-center justify-center px-5 py-2.5 font-medium rounded-lg bg-transparent text-ctp-text hover:bg-ctp-surface0 transition-colors"
          onclick={() => {
            goto("/signup");
          }}
        >
          Sign Up
        </button>
      </div>
    </div>
  </div>
</div>
