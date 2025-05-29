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
    feature1Title: string;
    feature1Desc: string;
    feature2Title: string;
    feature2Desc: string;
    feature3Title: string;
    feature3Desc: string;
    feature4Title: string;
    feature4Desc: string;
  };

  let currentLang: LangKey = $state("en");

  const copy: Record<LangKey, CopyContent> = {
    en: {
      line1: "Maximize Model Performance & Efficiency.",
      line2: "Tora: Visualize, Analyze, and Optimize Your ML Experiments.",
      line3: "Rapid Setup, Immediate Insights.",
      line4: "Automate & Accelerate Your MLOps with Tora.",
      getStarted: "Explore Tora",
      feature1Title: "Instant Integration, Immediate Value",
      feature1Desc:
        "Skip the hassle. Integrate Tora in minutes and start seeing results. Its design works with your existing tools, guaranteeing a smooth, zero-downtime transition.",
      feature2Title: "Dive into Data with Stunning Visuals!",
      feature2Desc:
        "Experience your data like never before! Interactive charts and real-time metrics bring your experiments to life, making it easy to spot trends and make data-driven decisions.",
      feature3Title: "AI: Your Strategic Advantage",
      feature3Desc:
        "Harness the power of AI to optimize your models. Tora's intelligent analysis provides key insights, helping you fine-tune parameters, detect anomalies, and maximize performance.",
      feature4Title: "MLOps: Together, Smarter",
      feature4Desc:
        "Build a thriving MLOps community. Tora's collaborative workspaces empower your team to share knowledge, manage experiments, and achieve more, together.",
    },
    ja: {
      line1: "モデルのパフォーマンスと効率を最大化",
      line2: "Tora: ML実験を可視化、分析、最適化",
      line3: "迅速なセットアップ、即座に洞察",
      line4: "ToraでMLOpsを自動化し、加速",
      getStarted: "Toraを探索",
      feature1Title: "シームレスな統合",
      feature1Desc:
        "Toraを数分で導入し、チームのワークフローを即座に加速します。",
      feature2Title: "実用的な可視化",
      feature2Desc:
        "複雑なモデルの挙動を明確で戦略的な洞察に変換し、迅速な意思決定を可能にします。",
      feature3Title: "AIによる診断",
      feature3Desc:
        "インテリジェントな分析を活用し、ボトルネックを積極的に特定し、モデルのパフォーマンスを最適化します。",
      feature4Title: "協調的なMLOps",
      feature4Desc:
        "実験データを一元管理し、レポート作成を合理化し、組織全体の知識共有を強化します。",
    },
  };

  function toggleLang() {
    currentLang = currentLang === "en" ? "ja" : "en";
  }

  const activeCopy = $derived(copy[currentLang]);
</script>

{#snippet FeatureCard(title: String, content: String)}
  <div
    class="opacity-75 backdrop-blur-md border border-ctp-overlay0/40 rounded-lg bg-gradient-to-br from-ctp-surface0/20 to-ctp-surface1/10 hover:from-ctp-surface0/40 hover:to-ctp-surface1/20 hover:border-ctp-lavender/30 hover:opacity-90 hover:scale-105 hover:shadow-xl transition-all duration-300 ease-out p-4 shadow-md shadow-ctp-overlay0/10"
  >
    <span class="font-bold text-ctp-lavender">{title}</span>
    <br />
    <span class="italic text-ctp-subtext0">
      {content}
    </span>
  </div>
{/snippet}

<Starfield />

<div class="absolute top-4 left-4 z-20 flex items-center space-x-2">
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
      class="relative z-10 flex w-full items-center justify-between px-2 text-xs font-medium"
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

<!-- Desktop Version -->
<section class="hidden md:flex flex-col h-full w-full text-ctp-text/95">
  <div class="h-2/3 flex flex-col">
    <div class="flex-none p-2">
      <div class="w-[clamp(31rem,68vw,73rem)] fill-ctp-blue/80">
        <Logo />
      </div>
    </div>

    <div class="flex-1 flex justify-end p-2">
      <div class="flex flex-col p-4 justify-center">
        <p class="text-base leading-relaxed">{activeCopy.line1}</p>
        <p class="text-base leading-relaxed mt-2">{activeCopy.line2}</p>
        <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
          {activeCopy.line3}
        </p>
        <p class="text-base leading-relaxed mt-2">
          {activeCopy.line4}
        </p>
        <button
          type="button"
          onclick={() => goto("/signup")}
          class="mt-4 p-2 rounded-lg bg-gradient-to-r from-ctp-blue/10 to-ctp-mauve/10 hover:from-ctp-blue/80 hover:to-ctp-mauve/80 hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md w-auto"
        >
          {activeCopy.getStarted}
        </button>
      </div>
    </div>
  </div>

  <div class="h-1/3 flex flex-col justify-center items-center">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(activeCopy.feature1Title, activeCopy.feature1Desc)}
      {@render FeatureCard(activeCopy.feature2Title, activeCopy.feature2Desc)}
      {@render FeatureCard(activeCopy.feature3Title, activeCopy.feature3Desc)}
      {@render FeatureCard(activeCopy.feature4Title, activeCopy.feature4Desc)}
    </div>
  </div>
</section>

<!-- Mobile Version -->
<section class="md:hidden flex flex-col h-full w-full text-ctp-text/95 px-4">
  <div class="flex flex-col space-y-8 pt-4">
    <div class="w-[clamp(16rem,85vw,32rem)] fill-ctp-blue/80">
      <Logo />
    </div>
    <div class="p-4">
      <p class="text-base leading-relaxed">{activeCopy.line1}</p>
      <p class="text-base leading-relaxed mt-2">{activeCopy.line2}</p>
      <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
        {activeCopy.line3}
      </p>
      <p class="text-base leading-relaxed mt-2">
        {activeCopy.line4}
      </p>
      <button
        type="button"
        onclick={() => goto("/signup")}
        class="mt-4 p-2 rounded-lg bg-gradient-to-r from-ctp-blue/10 to-ctp-mauve/10 hover:from-ctp-blue hover:to-ctp-mauve hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md w-auto"
      >
        {activeCopy.getStarted}
      </button>
    </div>
  </div>

  <div class="flex-1 pt-10">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(activeCopy.feature1Title, activeCopy.feature1Desc)}
      {@render FeatureCard(activeCopy.feature2Title, activeCopy.feature2Desc)}
      {@render FeatureCard(activeCopy.feature3Title, activeCopy.feature3Desc)}
      {@render FeatureCard(activeCopy.feature4Title, activeCopy.feature4Desc)}
    </div>
  </div>
</section>
