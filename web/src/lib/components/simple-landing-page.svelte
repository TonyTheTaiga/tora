<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "./logo_assets/logo.svelte";
  import { onMount } from "svelte";
  import { createHighlighter } from "shiki";
  import { browser } from "$app/environment";

  let activeTab: "install" | "code" = "install";

  const headline = "Pure Speed. Pure Insight. Zero Overhead.";
  const CTA = "start logging";

  const codeExample = `from tora import setup, tlog

setup("hello, world!")

# Log metrics
tlog("precision", 0.92)
tlog("recall", 0.76)`;

  const installationCommand = "pip install tora==0.0.2";

  let highlightedCode = "";
  let highlightedInstall = "";
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

  const formattedCode = addLineNumbers(codeExample);

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

      highlightedCode = highlighter.codeToHtml(codeExample, {
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
      });

      const shellCommand = `$ ${installationCommand}`;
      highlightedInstall = highlighter.codeToHtml(shellCommand, {
        lang: "bash",
        theme: currentTheme,
      });
    } catch (error) {
      console.error("Shiki highlighting failed:", error);
      highlightedCode = `<pre class="text-ctp-text font-mono"><code>${formattedCode}</code></pre>`;
      highlightedInstall = `<pre class="text-ctp-text font-mono"><code>$ ${installationCommand}</code></pre>`;
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
            class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-4 sm:mb-6 text-ctp-text"
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
                class="flex-1 px-4 py-2 text-xs font-mono transition-[background-color,opacity] duration-150 border-r border-ctp-surface0"
                class:bg-ctp-surface0={activeTab === "install"}
                class:opacity-50={activeTab !== "install"}
                onclick={() => (activeTab = "install")}
              >
                install
              </button>
              <button
                type="button"
                class="flex-1 px-4 py-2 text-xs font-mono transition-[background-color,opacity] duration-150 border-r border-ctp-surface0"
                class:bg-ctp-surface0={activeTab === "code"}
                class:opacity-50={activeTab !== "code"}
                onclick={() => (activeTab = "code")}
              >
                code
              </button>
            </div>

            <div class="p-4 sm:p-6 min-h-[180px] sm:min-h-[240px]">
              {#if activeTab === "code"}
                {#if highlightedCode}
                  <div
                    class="text-xs sm:text-sm md:text-base leading-relaxed overflow-x-auto text-left [&_pre]:!bg-transparent [&_code]:!bg-transparent"
                  >
                    {@html highlightedCode}
                  </div>
                {:else}
                  <pre
                    class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed overflow-x-auto text-left"><code
                      class="language-python">{@html formattedCode}</code
                    ></pre>
                {/if}
              {:else if activeTab === "install"}
                {#if highlightedInstall}
                  <div
                    class="text-xs sm:text-sm md:text-base leading-relaxed overflow-x-auto text-left [&_pre]:!bg-transparent [&_code]:!bg-transparent"
                  >
                    {@html highlightedInstall}
                  </div>
                {:else}
                  <pre
                    class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed overflow-x-auto text-left"><code
                      >$ {installationCommand}</code
                    ></pre>
                {/if}
              {/if}
            </div>

            <footer
              class="flex flex-col sm:flex-row justify-between items-center p-4 border-t border-ctp-surface0 gap-4"
            >
              <span class="text-xs text-ctp-subtext0 font-mono"
                >no accounts or CC required</span
              >
              <button
                type="button"
                onclick={() => goto("/signup")}
                class="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2 bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all font-mono text-sm"
              >
                {CTA}
              </button>
            </footer>
          </div>
        </article>
      </div>
    </div>
  </div>
</div>
