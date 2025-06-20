<script lang="ts">
  import { goto } from "$app/navigation";
  import Starfield from "./starfield.svelte";
  import Logo from "./logo.svelte";

  type LangKey = "en" | "ja";
  type CopyContent = {
    line1: string;
    line3: string;
    line4: string;
    CTA: string;
    headline: string;
    header1: string;
    header3: string;
    header4: string;
  };

  let currentLang: LangKey = $state("en");

  const codeExample = `from tora import log

# Track your experiments
log("loss", 0.045)
log("accuracy", 0.92)

# That's it. No setup required.`;

  function addLineNumbers(code: string): string {
    return code
      .split("\n")
      .map((line, index) => {
        const lineNum = (index + 1).toString().padStart(2, " ");
        return `<span class="text-ctp-overlay0 select-none">${lineNum}</span>  ${line}`;
      })
      .join("\n");
  }

  const formattedCode = $derived(addLineNumbers(codeExample));

  const copy: Record<LangKey, CopyContent> = {
    en: {
      line1: "Your command center for watching the loss go down.",
      line3: "Integrate in seconds. Sign up when you feel like it. Or don't.",
      line4: "Turn chaos into insights with zero configuration.",
      CTA: "Start Tracking",
      headline: "The Experiment Tracker You'll Actually Use.",
      header1: "Hypnotic Visuals",
      header3: "Zero Commitment",
      header4: "Chaos, Organized",
    },
    ja: {
      line1: "損失が下がるのを眺める、あなたの司令塔。",
      line3: "導入は数秒。登録は、気が向いたら。しなくてもOK。",
      line4: "設定不要で混沌を洞察に変える。",
      CTA: "トラッキング開始",
      headline: "あなたが、実際に使う実験トラッカー。",
      header1: "魅惑的なビジュアル",
      header3: "一切の縛りなし",
      header4: "混沌から秩序へ",
    },
  };

  function toggleLang() {
    currentLang = currentLang === "en" ? "ja" : "en";
  }

  const activeCopy = $derived.by(() => copy[currentLang]);
</script>

<Starfield />

<!-- Language Toggle Button -->
<div
  class="absolute top-4 right-4 z-20 flex items-center space-x-2 opacity-75 hover:opacity-100 transition-opacity duration-300 ease-in-out"
>
  <button
    type="button"
    onclick={toggleLang}
    class="relative inline-flex h-8 w-16 items-center rounded-full bg-ctp-surface0/80 backdrop-blur-sm p-0.5 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-ctp-blue focus:ring-offset-2 focus:ring-offset-ctp-base"
  >
    <span class="sr-only">Toggle language</span>
    <span
      class={`absolute left-0.5 top-0.5 h-7 w-7 rounded-full bg-ctp-blue/90 transition-transform duration-200 ease-in-out ${currentLang === "en" ? "translate-x-0" : "translate-x-8"}`}
      aria-hidden="true"
    ></span>
    <span
      class="relative z-10 flex w-full items-center justify-between px-2 text-xs font-mono font-medium"
    >
      <span
        class={`${currentLang === "en" ? "text-ctp-crust" : "text-ctp-subtext1"}`}
        >EN</span
      >
      <span
        class={`${currentLang === "ja" ? "text-ctp-crust" : "text-ctp-subtext1"}`}
        >JP</span
      >
    </span>
  </button>
</div>

<section class="min-h-screen flex items-center justify-center">
  <div class="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex flex-col items-center text-center text-ctp-text font-mono">
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
            {activeCopy.headline}
          </h1>
          <div class="w-16 sm:w-24 h-0.5 bg-ctp-blue mx-auto"></div>
        </div>

        <!-- Code Showcase - Prominent Position -->
        <div class="w-full max-w-3xl mx-auto">
          <div
            class="bg-ctp-surface0/50 border border-ctp-surface1 rounded-lg p-4 sm:p-6 backdrop-blur-sm shadow-2xl"
          >
            <div class="flex items-center justify-between mb-4">
              <div class="flex space-x-2">
                <div class="w-3 h-3 rounded-full bg-ctp-red"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-yellow"></div>
                <div class="w-3 h-3 rounded-full bg-ctp-green"></div>
              </div>
              <span class="text-xs text-ctp-subtext1 font-mono hidden sm:inline"
                >quick_start.py</span
              >
            </div>
            <div class="overflow-hidden">
              <pre
                class="text-xs sm:text-sm md:text-base text-ctp-text font-mono leading-relaxed overflow-x-auto text-left"><code
                  class="language-python">{@html formattedCode}</code
                ></pre>
            </div>
          </div>
        </div>

        <!-- Feature Grid -->
        <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
          <div
            class="p-4 sm:p-6 bg-ctp-surface0/20 rounded-lg backdrop-blur-sm border border-ctp-surface1/50"
          >
            <h2
              class="text-lg sm:text-xl md:text-2xl font-semibold mb-3 sm:mb-4 text-ctp-sapphire"
            >
              {activeCopy.header1}
            </h2>
            <p
              class="text-sm sm:text-base md:text-lg leading-relaxed text-ctp-subtext0"
            >
              {activeCopy.line1}
            </p>
          </div>

          <div
            class="p-4 sm:p-6 bg-ctp-surface0/20 rounded-lg backdrop-blur-sm border border-ctp-surface1/50"
          >
            <h2
              class="text-lg sm:text-xl md:text-2xl font-semibold mb-3 sm:mb-4 text-ctp-teal"
            >
              {activeCopy.header3}
            </h2>
            <p
              class="text-sm sm:text-base md:text-lg leading-relaxed text-ctp-subtext0"
            >
              {activeCopy.line3}
            </p>
          </div>

          <div
            class="p-4 sm:p-6 bg-ctp-surface0/20 rounded-lg backdrop-blur-sm border border-ctp-surface1/50 sm:col-span-2 lg:col-span-1"
          >
            <h3
              class="text-lg sm:text-xl md:text-2xl font-semibold mb-3 sm:mb-4 text-ctp-lavender"
            >
              {activeCopy.header4}
            </h3>
            <p
              class="text-sm sm:text-base md:text-lg leading-relaxed text-ctp-subtext0"
            >
              {activeCopy.line4}
            </p>
          </div>
        </div>

        <!-- Call to Action -->
        <div class="pt-4 sm:pt-8">
          <button
            type="button"
            onclick={() => goto("/signup")}
            class="w-full sm:w-auto px-8 sm:px-12 py-3 sm:py-4 text-base sm:text-lg md:text-xl font-semibold bg-ctp-blue/20 text-ctp-text border-2 border-ctp-blue/60 hover:bg-ctp-blue/30 hover:border-ctp-blue/80 transition-all duration-200 rounded-lg backdrop-blur-sm shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            {activeCopy.CTA}
          </button>
        </div>
      </div>
    </div>
  </div>
</section>
