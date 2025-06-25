<script lang="ts">
  import { goto } from "$app/navigation";
  import Logo from "./logo_assets/logo.svelte";

  const headline = "Pure Speed. Pure Insight. Zero Overhead.";
  const CTA = "Start Logging";

  const codeExample = `from tora import setup, tlog

setup("hello, world!")

# Log metrics
tlog("precision", 0.92)
tlog("recall", 0.76)`;

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
</script>

<div
  class="flex items-center justify-center min-h-[calc(100vh-2rem)] font-mono"
>
  <div class="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex flex-col items-center text-center text-ctp-text">
      <!-- Logo Section -->
      <div class="fill-ctp-blue w-full max-w-xs sm:max-w-sm mb-8 sm:mb-12">
        <Logo />
      </div>

      <!-- Hero Content -->
      <div class="w-full max-w-4xl space-y-8 sm:space-y-12">
        <!-- Headline -->
        <div>
          <h1
            class="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-4 sm:mb-6 text-ctp-text"
          >
            {headline}
          </h1>
          <div class="w-16 sm:w-24 h-0.5 bg-ctp-blue mx-auto"></div>
        </div>

        <!-- Code Showcase -->
        <div class="w-full max-w-3xl mx-auto">
          <div
            class="bg-ctp-surface0/10 backdrop-blur-md border border-ctp-surface0/20 overflow-hidden"
          >
            <div
              class="flex items-center justify-between px-4 py-3 border-b border-ctp-surface0"
            >
              <div class="flex space-x-2">
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-blue"></div>
              </div>
              <span class="text-xs text-ctp-subtext0 font-mono hidden sm:inline"
                >hw.py</span
              >
            </div>
            <div class="p-4 sm:p-6">
              <pre
                class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed overflow-x-auto text-left"><code
                  class="language-python">{@html formattedCode}</code
                ></pre>
            </div>

            <div class="flex justify-end p-4 border-t border-ctp-surface0">
              <button
                type="button"
                onclick={() => goto("/signup")}
                class="px-6 py-2 bg-ctp-blue/20 border border-ctp-blue/40 text-ctp-blue hover:bg-ctp-blue hover:text-ctp-crust transition-all font-mono text-sm"
              >
                {CTA}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
