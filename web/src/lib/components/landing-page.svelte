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
      line1: "Optimize Model Performance & Computational Efficiency.",
      line2: "Tora: Visualize, Analyze, and Optimize ML Experiment Workflows.",
      line3: "Fast Integration, Immediate Technical Insights.",
      line4: "Streamline MLOps Pipelines with Automated Tooling.",
      getStarted: "Explore Tora",
      feature1Title: "Seamless Tool Integration",
      feature1Desc:
        "Deploy in minutes with zero downtime. Integrates with existing ML stacks via Python client. Compatible with CI/CD pipelines and standard experiment frameworks.",
      feature2Title: "Advanced Data Visualization",
      feature2Desc:
        "Interactive dashboards display real-time metrics and experiment comparisons. Supports custom chart configurations and anomaly highlighting for efficient trend analysis.",
      feature3Title: "AI-Powered Experiment Analysis",
      feature3Desc:
        "Automated hyperparameter optimization suggestions based on experiment history. Identifies performance bottlenecks and generates targeted improvement recommendations.",
      feature4Title: "Collaborative MLOps Framework",
      feature4Desc:
        "Centralized experiment management with version control integration. Supports reproducibility through detailed metadata tracking and shared workspaces for distributed teams.",
    },
    ja: {
      line1: "モデル性能と計算効率の最適化",
      line2: "Tora: ML実験の可視化・分析・最適化プラットフォーム",
      line3: "迅速な導入、即時の洞察",
      line4: "MLOpsワークフローの自動化と高速化",
      getStarted: "Toraを試す",
      feature1Title: "高速統合と即時価値提供",
      feature1Desc:
        "数分で導入可能。既存ツールとシームレスに連携し、ダウンタイムなしでワークフローを強化します。CI/CDパイプラインへの統合も容易です。",
      feature2Title: "高度なデータ可視化",
      feature2Desc:
        "インタラクティブなチャートとリアルタイムメトリクスにより、複雑な実験データを直感的に理解。異常検出や傾向分析を効率化し、データ駆動型の意思決定をサポートします。",
      feature3Title: "AI支援の実験最適化",
      feature3Desc:
        "AIによる実験分析でハイパーパラメータ最適化を支援。異常値やボトルネックを自動検出し、モデルパフォーマンス向上のための具体的な提案を生成します。",
      feature4Title: "チーム協働MLOps",
      feature4Desc:
        "分散チームの知識を統合し、実験管理を効率化。バージョン管理、変更追跡、再現性確保の機能により、チーム全体の生産性と実験品質を向上させます。",
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
<section
  class="hidden md:flex flex-col min-h-[calc(100vh-2rem)] text-ctp-text/95"
>
  <div class="flex-[2] flex flex-col">
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

  <div class="flex-[1] flex flex-col justify-center items-center">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(activeCopy.feature1Title, activeCopy.feature1Desc)}
      {@render FeatureCard(activeCopy.feature2Title, activeCopy.feature2Desc)}
      {@render FeatureCard(activeCopy.feature3Title, activeCopy.feature3Desc)}
      {@render FeatureCard(activeCopy.feature4Title, activeCopy.feature4Desc)}
    </div>
  </div>
</section>

<!-- Mobile Version -->
<section
  class="md:hidden flex flex-col min-h-[calc(100vh-2rem)] text-ctp-text/95 px-4"
>
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
