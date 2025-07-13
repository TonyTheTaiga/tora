<script lang="ts">
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
  <button
    onclick={onCancel}
    type="button"
    class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-lavender hover:bg-ctp-lavender/10 hover:border-ctp-lavender/30 hover:text-ctp-lavender px-3 py-2 text-sm transition-all"
  >
    {cancelText}
  </button>
  <button
    type="submit"
    disabled={isSubmitting}
    class="bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30 px-3 py-2 text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed"
  >
    {submitText}
  </button>
</div>
