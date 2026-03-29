<script lang="ts">
  import { Button } from "$lib/components";

  let {
    onCancel,
    cancelText = "cancel",
    submitText = "submit",
    isSubmitting = false,
  }: {
    onCancel: () => void;
    cancelText?: string;
    submitText?: string;
    isSubmitting?: boolean;
  } = $props();
</script>

<div class="flex justify-end gap-2 pt-3 mt-3 border-t border-ctp-surface0/20">
  <Button onclick={onCancel} type="button">
    {cancelText}
  </Button>
  <Button type="submit" variant="primary" loading={isSubmitting}>
    {submitText}
  </Button>
</div>
