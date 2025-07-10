<script lang="ts">
  import { goto } from "$app/navigation";
  import { onMount } from "svelte";

  onMount(() => {
    setTimeout(() => {
      goto("/");
    }, 5000);
  });
</script>

<div class="h-full flex justify-center items-center text-center">
  <p>Email confirmed, redirecting the home page...</p>
</div>
