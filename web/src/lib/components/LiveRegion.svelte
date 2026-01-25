<script lang="ts">
  let {
    message,
    politeness = "polite",
  }: {
    message: string;
    politeness?: "polite" | "assertive";
  } = $props();
</script>

<div class="sr-only" role="status" aria-live={politeness} aria-atomic="true">
  {message}
</div>

<style>
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
