<script lang="ts">
  import { goto } from "$app/navigation";
  import Starfield from "./starfield.svelte";
  import Logo from "./logo.svelte"; // Import the Logo component

  type LangKey = 'en' | 'ja';
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

  let currentLang: LangKey = $state('en');

  const copy: Record<LangKey, CopyContent> = {
    en: {
      line1: "The path of complexity ends here.",
      line2: "Tora. The blade that reveals truth in data.",
      line3: "Three lines of code. Unsheathe its power.",
      line4: "Master your model. Discover Tora.",
      getStarted: "Get Started →",
      feature1Title: "Instant Integration",
      feature1Desc: "Effortless setup. Start seeing results in seconds.",
      feature2Title: "Insightful Visualization",
      feature2Desc: "Watch your experiments come to life with simple, beautiful charts.",
      feature3Title: "AI That Thinks With You",
      feature3Desc: "Tora’s AI delivers insights, not just data. So you can make smarter decisions, faster.",
      feature4Title: "Effortless Teamwork",
      feature4Desc: "Work together, share results, and keep everyone on the same page."
    },
    ja: {
      line1: "複雑さの道は、ここに終わる。",
      line2: "Tora。データに真実を現す刃。",
      line3: "コード三行。その力を解き放て。",
      line4: "モデルを極めよ。Toraを見出せ。",
      getStarted: "始める →",
      feature1Title: "瞬時の統合",
      feature1Desc: "簡単なセットアップ。数秒で結果を確認。",
      feature2Title: "洞察に満ちた可視化",
      feature2Desc: "シンプルで美しいチャートで、実験が生き生きと動き出すのをご覧ください。",
      feature3Title: "共に考えるAI",
      feature3Desc: "ToraのAIはデータだけでなく、洞察を提供します。より賢明な意思決定を、より速く。",
      feature4Title: "楽なチームワーク",
      feature4Desc: "協力し、結果を共有し、全員が同じ認識を持つ。"
    }
  };

  function toggleLang() {
    currentLang = currentLang === 'en' ? 'ja' : 'en';
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

<div class="absolute top-4 left-4 z-20 flex items-center space-x-2">
  <button
    type="button"
    onclick={toggleLang}
    class="relative inline-flex h-6 w-20 items-center rounded-full bg-ctp-surface0 p-0.5 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-ctp-blue focus:ring-offset-2 focus:ring-offset-ctp-base"
  >
    <span class="sr-only">Toggle language</span>
    <span 
      class={`absolute left-0 top-0.5 h-5 w-10 rounded-full bg-ctp-blue/90 transition-transform duration-200 ease-in-out ${currentLang === 'en' ? 'translate-x-0' : 'translate-x-10'}`}
      aria-hidden="true"
    ></span>
    <span class="relative z-10 flex w-full items-center justify-between px-2 text-xs font-medium">
      <span class={`${currentLang === 'en' ? 'text-ctp-crust' : 'text-ctp-subtext1'}`}>EN</span>
      <span class={`${currentLang === 'ja' ? 'text-ctp-crust' : 'text-ctp-subtext1'}`}>JP</span>
    </span>
  </button>
</div>

<Starfield />

<!-- Desktop Version -->
<section
  class="hidden md:flex flex-col h-full w-full text-ctp-text/95"
>
  <div class="h-2/3 flex flex-col">
    <div class="flex-none p-2">
      <div
        class="w-[clamp(31rem,68vw,73rem)] fill-ctp-blue/80"
      >
        <Logo />
      </div>
    </div>

    <div class="flex-1 flex justify-end p-2">
      <div class="w-1/3 flex flex-col z-10 p-4 opacity-75 bg-gradient-to-br from-ctp-blue/5 to-ctp-mauve/5 rounded-lg shadow-lg justify-center">
        <p class="text-base leading-relaxed">{activeCopy.line1}</p>
        <p class="text-base leading-relaxed mt-2">{activeCopy.line2}</p>
        <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
          {activeCopy.line3}
        </p>
        <p class="text-base leading-relaxed mt-2">
          {activeCopy.line4}
        </p>
        <button
          class="pt-4 text-start"
          type="button"
          onclick={() => goto("/signup")}
        >
          <span
            class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/10 to-ctp-mauve/10 hover:from-ctp-blue/80 hover:to-ctp-mauve/80 hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md"
            >{activeCopy.getStarted}</span
          >
        </button>
      </div>
    </div>
  </div>

  <div class="h-1/3 flex flex-col justify-center items-center">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        activeCopy.feature1Title,
        activeCopy.feature1Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature2Title,
        activeCopy.feature2Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature3Title,
        activeCopy.feature3Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature4Title,
        activeCopy.feature4Desc,
      )}
    </div>
  </div>
</section>

<!-- Mobile Version -->
<section class="md:hidden flex flex-col h-full w-full text-ctp-text/95 px-4">
  <div class="flex flex-col space-y-8 pt-4">
    <div
      class="w-[clamp(16rem,85vw,32rem)] fill-ctp-blue/80"
    >
      <Logo />
    </div>
    <div class="p-4 rounded-lg bg-ctp-crust/10">
      <p class="text-base leading-relaxed">{activeCopy.line1}</p>
      <p class="text-base leading-relaxed mt-2">{activeCopy.line2}</p>
      <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
        {activeCopy.line3}
      </p>
      <p class="text-base leading-relaxed mt-2">
        {activeCopy.line4}
      </p>
      <button type="button" onclick={() => goto("/signup")} class="mt-4">
        <span
          class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/20 to-ctp-mauve/20 hover:from-ctp-blue hover:to-ctp-mauve hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md"
          >{activeCopy.getStarted}</span
        >
      </button>
    </div>
  </div>

  <div class="flex-1 pt-10">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(
        activeCopy.feature1Title,
        activeCopy.feature1Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature2Title,
        activeCopy.feature2Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature3Title,
        activeCopy.feature3Desc,
      )}
      {@render FeatureCard(
        activeCopy.feature4Title,
        activeCopy.feature4Desc,
      )}
    </div>
  </div>
</section>
