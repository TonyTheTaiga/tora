<script lang="ts">
  import { UserRound } from "lucide-svelte";
</script>

<div class="flex items-center justify-center h-full">
  <div
    class="rounded-lg bg-ctp-crust border-2 border-dashed border-ctp-subtext0 p-4"
  >
    <article class="flex flex-col h-42 w-48">
      <div class="pb-2">
        <div
          class="w-8 h-8 flex items-center justify-center rounded-full border border-dashed border-ctp-blue text-ctp-blue"
        >
          <UserRound size={16} />
        </div>
      </div>

      <!-- Content -->
      <div class="flex-grow flex flex-col">
        <h3 class="text-sm font-medium text-ctp-subtext0 mb-1">
          Create an Account
        </h3>
        <p class="text-xs text-ctp-subtext0 leading-relaxed">
          Sign up to start tracking your experiments and metrics or browse
          public experiments.
        </p>
      </div>

      <!-- Footer -->
      <div class="pt-auto">
        <a
          href="/auth"
          class="px-2.5 py-1 rounded-md border border-ctp-blue text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-colors font-medium text-xs"
        >
          Sign Up / Login
        </a>
      </div>
    </article>
  </div>
</div>
