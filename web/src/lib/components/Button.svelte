<script lang="ts">
  import { Loader2 } from "@lucide/svelte";
  import type { Snippet } from "svelte";
  import type { HTMLButtonAttributes } from "svelte/elements";

  type ButtonVariant =
    | "default"
    | "primary"
    | "destructive"
    | "ghost"
    | "icon"
    | "link"
    | "cta";
  type ButtonSize = "sm" | "md" | "lg" | "icon";

  interface Props extends HTMLButtonAttributes {
    variant?: ButtonVariant;
    size?: ButtonSize;
    full?: boolean;
    loading?: boolean;
    loadingText?: string;
    children?: Snippet;
  }

  let {
    variant = "default",
    size = "md",
    full = false,
    loading = false,
    loadingText,
    disabled,
    class: className = "",
    children,
    ...restProps
  }: Props = $props();

  const baseClasses =
    "inline-flex items-center justify-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed";

  const variantClasses: Record<ButtonVariant, string> = {
    default:
      "bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-subtext0 hover:bg-ctp-surface0/30 hover:text-ctp-text",
    primary:
      "bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-blue hover:bg-ctp-blue/10 hover:border-ctp-blue/30",
    destructive:
      "bg-ctp-surface0/20 border border-ctp-surface0/30 text-ctp-red hover:bg-ctp-red/10 hover:border-ctp-red/30",
    ghost: "bg-transparent text-ctp-text hover:bg-ctp-surface0",
    icon: "text-ctp-subtext0 hover:text-ctp-text",
    link: "text-ctp-blue hover:text-ctp-blue/80 bg-transparent border-none",
    cta: "bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust font-mono",
  };

  const sizeClasses: Record<ButtonSize, string> = {
    sm: "px-2 py-1.5 text-xs",
    md: "px-3 py-2 text-sm",
    lg: "px-4 py-3",
    icon: "p-2",
  };

  let classes = $derived(
    [
      baseClasses,
      variantClasses[variant],
      sizeClasses[size],
      full ? "w-full" : "",
      className,
    ]
      .filter(Boolean)
      .join(" "),
  );
</script>

<button class={classes} disabled={disabled || loading} {...restProps}>
  {#if loading}
    <Loader2 size={size === "lg" ? 18 : 14} class="animate-spin" />
    {#if loadingText}
      {loadingText}
    {:else if children}
      {@render children()}
    {/if}
  {:else if children}
    {@render children()}
  {/if}
</button>
