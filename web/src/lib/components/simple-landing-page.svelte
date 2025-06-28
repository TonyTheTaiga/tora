<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "./logo_assets/logo.svelte";
  import { onMount } from "svelte";
  import { createHighlighter } from "shiki";
  import { browser } from "$app/environment";
  import { marked } from "marked";
  import { gettingStartedContent, userGuide } from "$lib/content";

  let activeTab: "start" | "guide" = "guide";

  const headline = "Pure Speed. Pure Insight. A New Experiment Tracker.";

  let highlightedGettingStarted = "";
  let isHighlighting = false;

  function addLineNumbers(code: string): string {
    return code
      .split("\n")
      .map((line, index) => {
        const lineNum = (index + 1).toString().padStart(2, " ");
        return `<span class="text-ctp-overlay0 select-none">${lineNum}</span>  ${line}`;
      })
      .join("\n");
  }

  const formattedGettingStarted = addLineNumbers(gettingStartedContent);

  function getTheme() {
    if (!browser) return "catppuccin-mocha";

    const htmlElement = document.documentElement;
    if (htmlElement.classList.contains("light")) {
      return "catppuccin-latte";
    }
    if (htmlElement.classList.contains("dark")) {
      return "catppuccin-mocha";
    }

    if (
      window.matchMedia &&
      window.matchMedia("(prefers-color-scheme: light)").matches
    ) {
      return "catppuccin-latte";
    }

    return "catppuccin-mocha";
  }

  async function highlightCode() {
    if (isHighlighting) return;
    isHighlighting = true;

    try {
      const currentTheme = getTheme();

      const highlighter = await createHighlighter({
        themes: ["catppuccin-mocha", "catppuccin-latte"],
        langs: ["python", "bash"],
      });

      highlightedGettingStarted = highlighter.codeToHtml(
        gettingStartedContent,
        {
          lang: "bash",
          theme: currentTheme,
          transformers: [
            {
              line(node, line) {
                node.properties["data-line"] = line;
                node.children.unshift({
                  type: "element",
                  tagName: "span",
                  properties: {
                    class: "line-number",
                    style:
                      "color: var(--color-ctp-overlay0); user-select: none; margin-right: 1em; display: inline-block; width: 2ch; text-align: right;",
                  },
                  children: [
                    { type: "text", value: line.toString().padStart(2, " ") },
                  ],
                });
              },
            },
          ],
        },
      );
    } catch (error) {
      console.error("Shiki highlighting failed:", error);
      highlightedGettingStarted = `<pre class="text-ctp-text font-mono"><code>${formattedGettingStarted}</code></pre>`;
    }
  }

  onMount(() => {
    highlightCode();

    if (browser && window.matchMedia) {
      const mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
      const handleThemeChange = () => {
        isHighlighting = false;
        highlightCode();
      };

      mediaQuery.addEventListener("change", handleThemeChange);
      const observer = new MutationObserver(handleThemeChange);
      observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ["class"],
      });

      return () => {
        mediaQuery.removeEventListener("change", handleThemeChange);
        observer.disconnect();
      };
    }
  });
</script>

<div
  class="flex items-center justify-center min-h-[calc(100vh-2rem)] font-mono"
>
  <div class="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex flex-col items-center text-center text-ctp-text">
      <div class="fill-ctp-blue w-full max-w-xs sm:max-w-sm mb-8 sm:mb-12">
        <Logo />
      </div>

      <div class="w-full max-w-4xl space-y-8 sm:space-y-12">
        <div>
          <h1
            class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-4 sm:mb-6 text-ctp-text font-mono"
          >
            {headline}
          </h1>
          <div class="w-16 sm:w-24 h-0.5 bg-ctp-blue mx-auto"></div>
        </div>

        <article class="w-full max-w-3xl mx-auto">
          <div
            class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 overflow-hidden"
          >
            <header
              class="grid grid-cols-3 items-center px-4 py-2 border-b border-ctp-surface0"
            >
              <div class="flex space-x-2">
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
              </div>
              <div
                class="text-center text-xs text-ctp-subtext0 font-mono hidden sm:inline"
              >
                ~/tora
              </div>
              <div></div>
            </header>
            <div class="flex border-b border-ctp-surface0">
              <button
                type="button"
                class="flex-1 px-4 py-2 text-xs font-mono"
                class:bg-ctp-surface0={activeTab === "start"}
                class:opacity-50={activeTab !== "start"}
                onclick={() => (activeTab = "start")}
              >
                quick_start.py
              </button>
              <button
                type="button"
                class="flex-1 px-4 py-2 text-xs font-mono"
                class:bg-ctp-surface0={activeTab === "guide"}
                class:opacity-50={activeTab !== "guide"}
                onclick={() => (activeTab = "guide")}
              >
                README.md
              </button>
            </div>

            <div
              class="p-4 sm:p-6 max-h-[220px] sm:min-h-[320px] overflow-y-auto"
            >
              {#if activeTab === "start"}
                {#if highlightedGettingStarted}
                  <div
                    class="text-xs sm:text-sm md:text-base leading-relaxed overflow-x-auto text-left [&_pre]:!bg-transparent [&_code]:!bg-transparent"
                  >
                    {@html highlightedGettingStarted}
                  </div>
                {:else}
                  <pre
                    class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed overflow-x-auto text-left"><code
                      >{@html formattedGettingStarted}</code
                    ></pre>
                {/if}
              {:else if activeTab === "guide"}
                <div class="markdown-content">
                  {@html marked(userGuide)}
                </div>
              {/if}
            </div>

            <footer
              class="flex flex-col sm:flex-row justify-between items-center p-4 border-t border-ctp-surface0 gap-4"
            >
              <span class="text-xs text-ctp-subtext0 font-mono"
                >start anonymous • sign up to store experiments</span
              >
              <button
                type="button"
                onclick={() => goto("/signup")}
                class="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2 bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all font-mono text-sm"
              >
                sign up
              </button>
            </footer>
          </div>
        </article>
      </div>
    </div>
  </div>
</div>

<style lang="postcss">
  @reference "tailwindcss";

  .markdown-content {
    @apply text-xs sm:text-sm md:text-base leading-relaxed text-left;
    color: var(--color-ctp-text);
  }

  .markdown-content :global(h1) {
    @apply text-lg font-bold mb-4 font-mono;
    color: var(--color-ctp-text);
  }

  .markdown-content :global(h2) {
    @apply text-base font-bold mb-3 mt-6 font-mono;
    color: var(--color-ctp-blue);
  }

  .markdown-content :global(h3) {
    @apply text-sm font-bold mb-2 mt-4 font-mono;
    color: var(--color-ctp-mauve);
  }

  .markdown-content :global(p) {
    @apply mb-3;
    color: var(--color-ctp-text);
  }

  .markdown-content :global(ul) {
    @apply mb-3 pl-4;
  }

  .markdown-content :global(li) {
    @apply mb-1;
    color: var(--color-ctp-text);
  }

  .markdown-content :global(strong) {
    @apply font-bold;
    color: var(--color-ctp-text);
  }

  /* Special styling for feature list items */
  .markdown-content :global(li strong:first-child) {
    color: var(--color-ctp-mauve);
  }

  .markdown-content :global(em) {
    @apply italic;
    color: var(--color-ctp-subtext1);
  }

  .markdown-content :global(code) {
    @apply px-1 py-0.5 rounded font-mono text-xs;
    background-color: rgba(var(--color-ctp-surface0), 0.3);
    color: var(--color-ctp-green);
  }

  .markdown-content :global(hr) {
    @apply my-6;
    border-color: var(--color-ctp-surface0);
  }

  .markdown-content :global(blockquote) {
    @apply border-l-2 pl-4 italic;
    border-color: var(--color-ctp-blue);
    color: var(--color-ctp-subtext1);
  }
</style>
