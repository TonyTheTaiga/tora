<script lang="ts">
  let {
    name,
    type = "text",
    placeholder,
    value = $bindable(),
    required = false,
    rows,
    error,
    validate,
    minlength,
    maxlength,
    ...restProps
  }: {
    name: string;
    type?: string;
    placeholder?: string;
    value?: string;
    required?: boolean;
    rows?: number;
    error?: string;
    validate?: (value: string) => string | null;
    minlength?: number;
    maxlength?: number;
    [key: string]: any;
  } = $props();

  let touched = $state(false);
  let internalError = $state<string | null>(null);

  // Compute displayed error
  let displayError = $derived(error || (touched ? internalError : null));

  function handleBlur() {
    touched = true;
    if (validate && value !== undefined) {
      internalError = validate(value);
    }
  }

  function handleInput(e: Event) {
    const target = e.target as HTMLInputElement | HTMLTextAreaElement;
    value = target.value;

    if (touched && validate) {
      internalError = validate(value);
    }
  }

  const baseClasses =
    "w-full bg-ctp-surface0/20 border px-3 py-2 text-ctp-text placeholder-ctp-subtext0 focus:outline-none focus:ring-1 transition-all text-sm";

  let inputClasses = $derived(
    displayError
      ? `${baseClasses} border-ctp-red/50 focus:ring-ctp-red focus:border-ctp-red`
      : `${baseClasses} border-ctp-surface0/30 focus:ring-ctp-blue focus:border-ctp-blue`,
  );
</script>

<div class="space-y-1">
  {#if type === "textarea"}
    <textarea
      {name}
      {placeholder}
      {required}
      {rows}
      {minlength}
      {maxlength}
      bind:value
      onblur={handleBlur}
      oninput={handleInput}
      class="{inputClasses} resize-none"
      aria-invalid={!!displayError}
      aria-describedby={displayError ? `${name}-error` : undefined}
      {...restProps}
    ></textarea>
  {:else}
    <input
      {name}
      {type}
      {placeholder}
      {required}
      {minlength}
      {maxlength}
      bind:value
      onblur={handleBlur}
      oninput={handleInput}
      class={inputClasses}
      aria-invalid={!!displayError}
      aria-describedby={displayError ? `${name}-error` : undefined}
      {...restProps}
    />
  {/if}

  {#if displayError}
    <p id="{name}-error" class="text-ctp-red text-xs" role="alert">
      {displayError}
    </p>
  {/if}
</div>
