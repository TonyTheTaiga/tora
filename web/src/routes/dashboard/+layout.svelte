<script lang="ts">
  import "../../app.css";
  import { getTheme } from "$lib/state/theme.svelte.js";

  let { children } = $props();
  let theme = $derived.by(() => getTheme());
</script>

<!-- Dashboard uses its own minimal layout without global header/toolbar -->
<div class="min-h-screen text-ctp-text surface-base" data-theme={theme}>
  {@render children()}
</div>
