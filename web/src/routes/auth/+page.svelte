<div class="flex items-center justify-center">
  <form
    class="text-ctp-text max-w-md p-8 rounded bg-ctp-base flex flex-col space-y-2"
    method="POST"
    autocomplete="on"
    action="?/login"
    name="login-form"
  >
    <label>
      Email
      <input
        name="email"
        type="email"
        class="bg-ctp-mantle px-2"
        autocomplete="email"
        required
      />
    </label>
    <label>
      Password
      <input
        name="password"
        type="password"
        minlength="6"
        class="bg-ctp-mantle px-2"
        autocomplete="current-password"
        required
      />
    </label>
    <div>
      <button type="submit" formaction="?/signup">Sign up</button>
      <button type="submit">Sign In</button>
    </div>
  </form>
</div>
