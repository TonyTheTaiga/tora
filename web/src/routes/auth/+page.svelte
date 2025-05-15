<div class="flex items-center justify-center min-h-screen">
  <form
    class="text-ctp-text max-w-md p-8 rounded bg-ctp-base flex flex-col space-y-2"
    method="POST"
    action="?/signup"
    autocomplete="on"
    name="login-form"
  >
    <label>
      Email
      <input name="email" type="email" class="bg-ctp-mantle px-2" autocomplete="username email" />
    </label>
    <label>
      Password
      <input name="password" type="password" class="bg-ctp-mantle px-2" autocomplete="current-password" />
    </label>
    <div>
      <button type="submit">Sign up</button>
      <button type="submit" formaction="?/signin">Sign In</button>
    </div>
  </form>
</div>