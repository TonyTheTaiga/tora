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

  const copy: Record<LangKey, CopyContent> = {
    en: {
      line1: "Your command center for watching the loss go down.",
      line3: "Integrate in seconds. Sign up when you feel like it. Or don't.",
      line4: "",
      CTA: "Run Your First Plot",
      headline: "The Experiment Tracker You'll Actually Use.",
      header1: "Hypnotic Visuals",
      header3: "Zero Commitment",
      header4: "Chaos, Organized",
    },
    ja: {
      line1: "損失が下がるのを眺める、あなたの司令塔。",
      line3: "導入は数秒。登録は、気が向いたら。しなくてもOK。",
      line4: "",
      CTA: "最初のプロットを実行",
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

<section class="min-h-screen flex items-center justify-center px-4">
  <div class="max-w-6xl mx-auto">
    <div class="flex flex-col items-center text-center text-ctp-text font-mono">
      <div class="fill-ctp-blue w-full max-w-sm mb-16">
        <Logo />
      </div>
      <div class="p-12 md:p-16 max-w-4xl">
        <div class="mb-12">
          <h1
            class="text-3xl md:text-4xl lg:text-5xl font-bold leading-tight mb-6 text-ctp-text"
          >
            {activeCopy.headline}
          </h1>
          <div class="w-24 h-0.5 bg-ctp-blue mx-auto mb-8"></div>
        </div>
        <div class="grid md:grid-cols-2 gap-8 mb-12">
          <div class="p-6">
            <h2
              class="text-xl md:text-2xl font-semibold mb-4 text-ctp-sapphire"
            >
              {activeCopy.header1}
            </h2>
            <p class="text-base md:text-lg leading-relaxed text-ctp-subtext0">
              {activeCopy.line1}
            </p>
          </div>
          <div class="p-6">
            <h2 class="text-xl md:text-2xl font-semibold mb-4 text-ctp-teal">
              {activeCopy.header3}
            </h2>
            <p class="text-base md:text-lg leading-relaxed text-ctp-subtext0">
              {activeCopy.line3}
            </p>
          </div>
        </div>
        <div class="mb-12">
          <div class="p-6">
            <h3 class="text-lg md:text-xl font-medium mb-3 text-ctp-lavender">
              {activeCopy.header4}
            </h3>
            <p class="text-base md:text-lg leading-relaxed text-ctp-subtext0">
              {activeCopy.line4}
            </p>
          </div>
        </div>
        <div class="pt-4">
          <button
            type="button"
            onclick={() => goto("/signup")}
            class="px-12 py-4 text-lg md:text-xl font-semibold bg-ctp-blue/20 text-ctp-text border-2 border-ctp-blue/60 hover:bg-ctp-blue/30 hover:border-ctp-blue/80 transition-colors duration-200 rounded-md backdrop-blur-sm"
          >
            {activeCopy.CTA}
          </button>
        </div>
      </div>
    </div>
  </div>
</section>
