<script lang="ts">
  import { goto } from "$app/navigation";
  import Starfield from "./starfield.svelte";
  import Logo from "./logo.svelte"; // Import the Logo component
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
      <div class="w-1/3 flex flex-col z-10 p-4">
        <p class="text-base leading-relaxed">Tired of the old way?</p>
        <p class="text-base leading-relaxed mt-2">Tora changes everything.</p>
        <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
          More discovery. Less busywork.
        </p>
        <p class="text-base leading-relaxed mt-2">
          It’s time for a breakthrough.
        </p>
        <button
          class="pt-4 text-start"
          type="button"
          onclick={() => goto("/signup")}
        >
          <span
            class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/10 to-ctp-mauve/10 hover:from-ctp-blue/80 hover:to-ctp-mauve/80 hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md"
            >Get Started →</span
          >
        </button>
      </div>
    </div>
  </div>

  <div class="h-1/3 flex flex-col justify-center items-center">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        "Instant Integration",
        "Effortless setup. Start seeing results in seconds.",
      )}
      {@render FeatureCard(
        "Insightful Visualization",
        "Watch your experiments come to life with simple, beautiful charts.",
      )}
      {@render FeatureCard(
        "AI That Thinks With You",
        "Tora’s AI delivers insights, not just data. So you can make smarter decisions, faster.",
      )}
      {@render FeatureCard(
        "Effortless Teamwork",
        "Work together, share results, and keep everyone on the same page.",
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
      <p class="text-base leading-relaxed">Tired of the old way?</p>
      <p class="text-base leading-relaxed mt-2">Tora changes everything.</p>
      <p class="text-base leading-relaxed mt-2 font-bold text-ctp-sapphire">
        More discovery. Less busywork.
      </p>
      <p class="text-base leading-relaxed mt-2">
        It’s time for a breakthrough.
      </p>
      <button type="button" onclick={() => goto("/signup")} class="mt-4">
        <span
          class="p-2 rounded-lg bg-gradient-to-r from-ctp-blue/20 to-ctp-mauve/20 hover:from-ctp-blue hover:to-ctp-mauve hover:scale-105 hover:shadow-lg transition-all duration-300 ease-out hover:text-ctp-crust font-medium border border-ctp-overlay0/30 hover:border-ctp-lavender/50 shadow-md"
          >Get Started →</span
        >
      </button>
    </div>
  </div>

  <div class="flex-1 pt-10">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(
        "Instant Integration",
        "Effortless setup. Start seeing results in seconds.",
      )}
      {@render FeatureCard(
        "Insightful Visualization",
        "Watch your experiments come to life with simple, beautiful charts.",
      )}
      {@render FeatureCard(
        "AI That Thinks With You",
        "Tora’s AI delivers insights, not just data. So you can make smarter decisions, faster.",
      )}
      {@render FeatureCard(
        "Effortless Teamwork",
        "Work together, share results, and keep everyone on the same page.",
      )}
    </div>
  </div>
</section>
