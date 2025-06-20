<script lang="ts">
  import { goto } from "$app/navigation";
  import Starfield from "./starfield.svelte";
  import Logo from "./logo.svelte";

  type LangKey = "en" | "ja";
  type CopyContent = {
    line1: string;
    line2: string;
    line3: string;
    line4: string;
    getStarted: string;
  };

  let currentLang: LangKey = $state("en");

  const copy: Record<LangKey, CopyContent> = {
    en: {
      line1: "Optimize Model Performance & Computational Efficiency.",
      line2: "Tora: Visualize, Analyze, and Optimize ML Experiment Workflows.",
      line3: "Fast Integration, Immediate Technical Insights.",
      line4: "Streamline MLOps Pipelines with Automated Tooling.",
      getStarted: "explore tora",
    },
    ja: {
      line1: "モデル性能と計算効率の最適化",
      line2: "Tora: ML実験の可視化・分析・最適化プラットフォーム",
      line3: "迅速な導入、即時の洞察",
      line4: "MLOpsワークフローの自動化と高速化",
      getStarted: "toraを試す",
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
  class="absolute top-4 left-4 z-20 flex items-center space-x-2 opacity-75 hover:opacity-100 transition-opacity duration-300 ease-in-out"
>
  <button
    type="button"
    onclick={toggleLang}
    class="relative inline-flex h-6 w-20 items-center rounded-full bg-ctp-surface0 p-0.5 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-ctp-blue focus:ring-offset-2 focus:ring-offset-ctp-base"
  >
    <span class="sr-only">Toggle language</span>
    <span
      class={`absolute left-0 top-0.5 h-5 w-9 rounded-full bg-ctp-blue/90 transition-transform duration-200 ease-in-out ${currentLang === "en" ? "translate-x-0.5" : "translate-x-[2.625rem]"}`}
      aria-hidden="true"
    ></span>
    <span
      class="relative z-10 flex w-full items-center justify-between px-2 text-xs font-mono"
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
  <div class="max-w-4xl mx-auto px-6 md:px-8 lg:px-12">
    <div class="flex flex-col items-center text-center text-ctp-text font-mono space-y-8">
      <div class="fill-ctp-blue w-full max-w-md">
        <Logo />
      </div>

      <div class="flex flex-col space-y-6 max-w-2xl">
        <p class="text-lg md:text-xl leading-relaxed">{activeCopy.line1}</p>
        <p class="text-lg md:text-xl leading-relaxed">{activeCopy.line2}</p>
        <p class="text-lg md:text-xl leading-relaxed text-ctp-sapphire">
          {activeCopy.line3}
        </p>
        <p class="text-lg md:text-xl leading-relaxed">
          {activeCopy.line4}
        </p>
        
        <div class="pt-4">
          <button
            type="button"
            onclick={() => goto("/signup")}
            class="px-8 py-4 text-lg bg-ctp-blue/20 border border-ctp-blue/40 hover:bg-ctp-blue/30 hover:border-ctp-blue/60 transition-all font-mono"
          >
            {activeCopy.getStarted}
          </button>
        </div>
      </div>
    </div>
  </div>
</section>
