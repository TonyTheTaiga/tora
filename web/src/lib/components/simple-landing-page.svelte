<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "$lib/logo_assets/logo.svelte";
  import { createHighlighter } from "shiki";
  import { browser } from "$app/environment";
  import { marked } from "marked";
  import { gettingStartedContent, userGuide } from "$lib/content";

  let activeTab: "start" | "guide" = $state<"start" | "guide">("guide");
  let isMaximized = $state(false);
  let highlightedGettingStarted = $state("");
  let isHighlighting = $state(false);

  $effect(() => {
    if (browser && document.body) {
      if (isMaximized) {
        document.body.style.overflow = "hidden";
      } else {
        document.body.style.overflow = "";
      }
    }

    return () => {
      if (browser && document.body) {
        document.body.style.overflow = "";
      }
    };
  });

  const headline = "Pure Speed. Pure Insight.";
  const subtitle = "A Modern Experiment Tracker.";

  function addLineNumbers(code: string): string {
    return code
      .trim()
      .split("\n")
      .filter((line) => line.trim() !== "")
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
        langs: ["python"],
      });

      highlightedGettingStarted = highlighter.codeToHtml(
        gettingStartedContent,
        {
          lang: "python",
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

  $effect(() => {
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
            class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-2 sm:mb-3 text-ctp-text font-mono"
          >
            {headline}
          </h1>
          <h2
            class="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold leading-tight mb-4 sm:mb-6 text-ctp-text font-mono"
          >
            {subtitle}
          </h2>
          <div class="w-16 sm:w-24 h-0.5 bg-ctp-blue mx-auto"></div>
        </div>

        <article
          class="w-full max-w-3xl mx-auto layer-fade-in"
          class:maximized={isMaximized}
        >
          <div
            class="terminal-chrome overflow-hidden stack-layer"
            class:maximized-terminal={isMaximized}
          >
            <header
              class="terminal-chrome-header grid grid-cols-3 items-center"
            >
              <div class="flex space-x-2">
                <div class="w-3 h-3 rounded-full bg-ctp-overlay0"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-overlay0"></div>
                <button
                  type="button"
                  onclick={() => (isMaximized = !isMaximized)}
                  class="w-3 h-3 rounded-full bg-ctp-blue cursor-pointer"
                  title={isMaximized ? "Restore" : "Maximize"}
                  aria-label={isMaximized
                    ? "Restore terminal window"
                    : "Maximize terminal window"}
                ></button>
              </div>
              <div
                class="text-center text-xs text-ctp-subtext0 font-mono hidden sm:inline"
              >
                ~/tora
              </div>
              <div></div>
            </header>
            <div class="flex relative">
              <button
                type="button"
                class="flex-1 px-4 py-2 text-xs font-mono relative transition-colors"
                class:bg-ctp-surface0={activeTab === "start"}
                class:text-ctp-text={activeTab === "start"}
                class:text-ctp-subtext0={activeTab !== "start"}
                class:hover:text-ctp-text={activeTab !== "start"}
                onclick={() => (activeTab = "start")}
              >
                quick_start.py
                {#if activeTab === "start"}
                  <div
                    class="absolute bottom-0 left-0 right-0 h-0.5 bg-ctp-blue"
                  ></div>
                {/if}
              </button>
              <button
                type="button"
                class="flex-1 px-4 py-2 text-xs font-mono relative transition-colors"
                class:bg-ctp-surface0={activeTab === "guide"}
                class:text-ctp-text={activeTab === "guide"}
                class:text-ctp-subtext0={activeTab !== "guide"}
                class:hover:text-ctp-text={activeTab !== "guide"}
                onclick={() => (activeTab = "guide")}
              >
                README.md
                {#if activeTab === "guide"}
                  <div
                    class="absolute bottom-0 left-0 right-0 h-0.5 bg-ctp-blue"
                  ></div>
                {/if}
              </button>
              <div
                class="absolute bottom-0 left-0 right-0 h-px bg-ctp-surface0/30"
              ></div>
            </div>

            <div
              class="p-4 sm:p-6 max-h-[220px] sm:min-h-[320px] overflow-y-auto"
              class:maximized-content={isMaximized}
            >
              {#if activeTab === "start"}
                {#if highlightedGettingStarted}
                  <div
                    class="text-xs sm:text-sm md:text-base leading-relaxed text-left [&_pre]:!bg-transparent [&_code]:!bg-transparent [&_pre]:whitespace-pre-wrap [&_pre]:break-words"
                  >
                    {@html highlightedGettingStarted}
                  </div>
                {:else}
                  <pre
                    class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed text-left whitespace-pre-wrap break-words"><code
                      >{@html formattedGettingStarted}</code
                    ></pre>
                {/if}
              {:else if activeTab === "guide"}
                <div class="markdown-content break-words">
                  {@html marked(userGuide)}
                </div>
              {/if}
            </div>

            <footer
              class="surface-layer-1 border-t border-ctp-surface0/30 flex flex-col sm:flex-row justify-between items-center p-4 gap-4"
            >
              <span class="text-xs text-ctp-subtext0 font-mono"
                >start anonymous • sign up to store experiments</span
              >
              <button
                type="button"
                onclick={() => goto("/signup")}
                class="w-full sm:w-auto text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust font-mono text-sm flex items-center justify-center gap-2 px-6 py-2 border border-ctp-blue"
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
    word-wrap: break-word;
    overflow-wrap: break-word;
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
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .markdown-content :global(pre) {
    @apply font-mono text-xs p-3 rounded mb-3;
    background-color: rgba(var(--color-ctp-surface0), 0.3);
    color: var(--color-ctp-text);
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }

  .markdown-content :global(pre code) {
    @apply p-0;
    background-color: transparent;
    white-space: pre-wrap;
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

  /* Maximized terminal styles */
  .maximized {
    @apply fixed inset-0 z-50 max-w-none w-full h-full m-0;
  }

  .maximized-terminal {
    @apply h-full;
  }

  .maximized-content {
    @apply max-h-none min-h-0;
    height: calc(100vh - 200px);
  }
</style>
