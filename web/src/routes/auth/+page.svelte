<script lang="ts">
  import { LogIn, User, Lock, Mail } from "lucide-svelte";
  let signUp = $state(false);
</script>

<div class="flex items-center justify-center min-h-[80vh]">
  <div class="w-full max-w-md">
    <div class="bg-ctp-base shadow-lg rounded-xl overflow-hidden border border-ctp-surface1">
      <!-- Header -->
      <div class="p-6 bg-gradient-to-r from-ctp-blue to-ctp-mauve">
        <h2 class="text-2xl font-semibold text-ctp-crust flex items-center gap-2">
          <User size={24} />
          {signUp ? 'Create Account' : 'Welcome Back'}
        </h2>
        <p class="text-ctp-crust/80 mt-1">
          {signUp ? 'Join Tora to track your experiments' : 'Sign in to your account'}
        </p>
      </div>
      
      <!-- Form -->
      <form
        method="POST"
        autocomplete="on"
        action={signUp ? '?/signup' : '?/login'}
        name="login-form"
        class="p-6 space-y-5"
      >
        <!-- Email field -->
        <div class="space-y-2">
          <label class="text-sm font-medium text-ctp-subtext0 block" for="email">
            Email Address
          </label>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Mail size={18} class="text-ctp-subtext0" />
            </div>
            <input
              id="email"
              name="email"
              type="email"
              autocomplete="email"
              required
              class="w-full pl-10 pr-4 py-3 bg-ctp-mantle rounded-lg border border-ctp-surface0 focus:outline-none focus:ring-2 focus:ring-ctp-blue text-ctp-text"
              placeholder="your.email@example.com"
            />
          </div>
        </div>
        
        <!-- Password field -->
        <div class="space-y-2">
          <label class="text-sm font-medium text-ctp-subtext0 block" for="password">
            Password
          </label>
          <div class="relative">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Lock size={18} class="text-ctp-subtext0" />
            </div>
            <input
              id="password"
              name="password"
              type="password"
              minlength="6"
              autocomplete={signUp ? "new-password" : "current-password"}
              required
              class="w-full pl-10 pr-4 py-3 bg-ctp-mantle rounded-lg border border-ctp-surface0 focus:outline-none focus:ring-2 focus:ring-ctp-mauve text-ctp-text"
              placeholder={signUp ? "Create a strong password" : "Enter your password"}
            />
          </div>
        </div>
        
        <!-- Button actions -->
        <div>
          <button
            type="submit"
            class="w-full flex items-center justify-center gap-2 rounded-lg bg-gradient-to-r from-ctp-blue to-ctp-mauve py-3 px-4 text-ctp-crust font-medium hover:shadow-lg transition-all"
          >
            <LogIn size={18} />
            {signUp ? 'Create Account' : 'Sign In'}
          </button>
        </div>
      </form>
      
      <!-- Footer -->
      <div class="px-6 py-4 bg-ctp-surface0/30 text-center">
        <p class="text-sm text-ctp-subtext0">
          {signUp ? 'Already have an account?' : "Don't have an account?"}
          <button
            type="button"
            class="text-ctp-blue hover:text-ctp-mauve font-medium ml-1"
            onclick={() => signUp = !signUp}
          >
            {signUp ? 'Sign In' : 'Sign Up'}
          </button>
        </p>
      </div>
    </div>
  </div>
</div>
