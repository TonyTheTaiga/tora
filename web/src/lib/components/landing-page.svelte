<script lang="ts">
  import { goto } from "$app/navigation";
  import Starfield from "./starfield.svelte";
</script>

{#snippet FeatureCard(title: String, content: String)}
  <div
    class="border border-ctp-overlay0/50 rounded-lg p-4 bg-gradient-to-br from-ctp-surface0/30 to-ctp-surface1/20 hover:from-ctp-surface0/50 hover:to-ctp-surface1/30 transition-all duration-300"
  >
    <span class="font-bold text-ctp-lavender">{title}</span>
    <br />
    <span class="italic text-ctp-subtext0">
      {content}
    </span>
  </div>
{/snippet}

<Starfield />

<section
  class="hidden md:flex flex-col h-full w-full text-ctp-text/90 relative"
>
  <div class="flex-none w-full p-2 relative">
    <span
      class="font-extrabold leading-none text-[clamp(8rem,20vw,32rem)] text-ctp-text"
    >
      TORA
    </span>
  </div>

  <div class="flex justify-end w-full p-2 relative">
    <div class="w-1/3 flex flex-col relative z-10">
      <p class="text-base leading-relaxed">
        Your experiments deserve <span class="font-bold">clarity</span>, not
        chaos.
      </p>
      <p class="text-base leading-relaxed mt-4">
        TORA streamlines experiment tracking, <span class="font-bold"
          >just three simple lines of Python.</span
        >
      </p>
      <p class="text-base leading-relaxed mt-4 font-bold text-ctp-sapphire">
        Less noise. More insight. Immediate impact.
      </p>
      <p class="text-base leading-relaxed mt-2">
        With distractions gone, your best work emerges naturally.
      </p>
      <button
        class="pt-8 text-start"
        type="button"
        onclick={() => goto("/signup")}
      >
        <span
          class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/20 to-ctp-mauve/20 hover:from-ctp-blue hover:to-ctp-mauve transition-all duration-300 hover:text-ctp-crust font-medium"
          >Start Seeing Clearly →</span
        >
      </button>
    </div>
  </div>

  <div class="flex-1 py-4">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        "Instant Setup",
        "Integrate effortlessly with three lines of Python—minimal disruption, maximum clarity.",
      )}
      {@render FeatureCard(
        "Focused Insights",
        "Clear visualizations and direct comparisons ensure you see what truly matters.",
      )}
      {@render FeatureCard(
        "Intuitive Support",
        "AI-powered suggestions gently guide your experiments toward optimal outcomes.",
      )}
      {@render FeatureCard(
        "Seamless Collaboration",
        "Share experiments effortlessly. Unified context leads to natural teamwork and reproducibility.",
      )}
    </div>
  </div>
</section>

<section class="md:hidden flex flex-col h-full w-full text-ctp-text/90 px-4">
  <div class="flex flex-col space-y-8 pt-4">
    <h1
      class="font-extrabold leading-none text-[clamp(6rem,30vw,14rem)] text-ctp-text"
    >
      TORA
    </h1>
    <p class="text-base leading-relaxed">
      Your experiments deserve <span class="font-bold">clarity</span>, not
      chaos.
    </p>
    <p class="text-base leading-relaxed mt-4">
      TORA streamlines experiment tracking, <span class="font-bold"
        >just three simple lines of Python.</span
      >
    </p>
    <p class="text-base leading-relaxed mt-4 font-bold text-ctp-sapphire">
      Less noise. More insight. Immediate impact.
    </p>
    <p class="text-base leading-relaxed mt-2">
      With distractions gone, your best work emerges naturally.
    </p>
    <button type="button" onclick={() => goto("/signup")}>
      <span
        class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/20 to-ctp-mauve/20 hover:from-ctp-blue hover:to-ctp-mauve transition-all duration-300 hover:text-ctp-crust font-medium"
        >Start Seeing Clearly →</span
      >
    </button>
  </div>

  <div class="flex-1 py-4">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(
        "Instant Setup",
        "Integrate effortlessly with three lines of Python—minimal disruption, maximum clarity.",
      )}
      {@render FeatureCard(
        "Focused Insights",
        "Clear visualizations and direct comparisons ensure you see what truly matters.",
      )}
      {@render FeatureCard(
        "Intuitive Support",
        "AI-powered suggestions gently guide your experiments toward optimal outcomes.",
      )}
      {@render FeatureCard(
        "Seamless Collaboration",
        "Share experiments effortlessly. Unified context leads to natural teamwork and reproducibility.",
      )}
    </div>
  </div>
</section>
