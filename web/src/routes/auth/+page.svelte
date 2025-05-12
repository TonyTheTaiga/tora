<div class="flex items-center justify-center min-h-screen">
  <form
    class="text-ctp-text max-w-md p-8 rounded bg-ctp-base flex flex-col space-y-2"
    method="POST"
    action="?/signup"
  >
    <label>
      Email
      <input name="email" type="email" class="bg-ctp-mantle px-2" />
    </label>
    <label>
      Password
      <input name="password" type="password" class="bg-ctp-mantle px-2" />
    </label>
    <div>
      <button>Sign up</button>
      <button formaction="?/signin">Sign In</button>
    </div>
  </form>
</div>
