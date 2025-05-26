<script lang="ts">
  import { goto } from "$app/navigation";
</script>

{#snippet FeatureCard(title: String, content: String)}
  <div class="border border-ctp-overlay0/50 rounded-lg p-4">
    <span class="font-bold">{title}</span>
    <br />
    <span class="italic">
      {content}
    </span>
  </div>
{/snippet}

<section class="hidden md:flex flex-col h-full w-full text-ctp-text/90">
  <div class="flex-none w-full p-2">
    <span class="font-extrabold leading-none text-[clamp(8rem,20vw,32rem)]">
      TORA
    </span>
  </div>

  <div class="flex justify-end w-full p-2">
    <div class="w-1/3 flex flex-col">
      <p class="text-base leading-relaxed">
        You build models. You seek breakthroughs. That's the work.
      </p>
      <p class="text-base leading-relaxed mt-4">
        But often, the process is filled with noise: experiment tracking,
        scattered details, the friction that pulls you from true insight.
      </p>
      <p class="text-base leading-relaxed mt-2 font-bold">
        Quiet the noise. Amplify insight. In three lines of code.
      </p>
      <p class="text-base leading-relaxed mt-4">
        Suddenly, your experiments are clear. Your focus, sharp. The essential
        truths of your work emerge, naturally.
      </p>
      <button
        class="pt-8 text-start"
        type="button"
        onclick={() => goto("/signup")}
      >
        <span
          class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
          >Find Your Focus →</span
        >
      </button>
    </div>
  </div>

  <div class="flex-1 py-4">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        "Pure Connection.",
        "Three lines of Python. TORA meets your workflow, any framework. No forced structures, just a direct line from your code to clarity.",
      )}
      {@render FeatureCard(
        "The Essential Story.",
        "Your data, presented with honesty. Clean visualizations, direct comparisons. See the true narrative of your experiments, without distraction.",
      )}
      {@render FeatureCard(
        "Intuitive Guidance.",
        "AI that serves your intuition. Gentle suggestions to explore and refine. Discover optimal paths, letting your model's unique voice emerge.",
      )}
      {@render FeatureCard(
        "Shared Resonance.",
        "Unite your team around the core work. Clear experiment histories, shared context. Build together with genuine understanding. Reproducibility, naturally.",
      )}
    </div>
  </div>
</section>

<section class="md:hidden flex flex-col h-full w-full text-ctp-text/90 px-4">
  <div class="flex flex-col space-y-8 pt-4">
    <h1 class="font-extrabold leading-none text-[clamp(6rem,30vw,14rem)]">
      TORA
    </h1>
    <p class="text-base leading-relaxed">
      You build models. You seek breakthroughs. That's the work.
    </p>
    <p class="text-base leading-relaxed mt-4">
      But often, the process is filled with noise: experiment tracking,
      scattered details, the friction that pulls you from true insight.
    </p>
    <p class="text-base leading-relaxed mt-4">
      TORA helps you find the **signal**.
    </p>
    <p class="text-base leading-relaxed mt-2 font-bold">
      Quiet the noise. Amplify insight. In three lines of code.
    </p>
    <p class="text-base leading-relaxed mt-4">
      Suddenly, your experiments are clear. Your focus, sharp. The essential
      truths of your work emerge, naturally.
    </p>
    <p class="text-base leading-relaxed mt-4">TORA. Get back to building.</p>
    <button type="button" onclick={() => goto("/signup")}>
      <span
        class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
        >Find Your Focus →</span
      >
    </button>
  </div>

  <div class="flex-1 py-4">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(
        "Pure Connection.",
        "Three lines of Python. TORA meets your workflow, any framework. No forced structures, just a direct line from your code to clarity.",
      )}
      {@render FeatureCard(
        "The Essential Story.",
        "Your data, presented with honesty. Clean visualizations, direct comparisons. See the true narrative of your experiments, without distraction.",
      )}
      {@render FeatureCard(
        "Intuitive Guidance.",
        "AI that serves your intuition. Gentle suggestions to explore and refine. Discover optimal paths, letting your model's unique voice emerge.",
      )}
      {@render FeatureCard(
        "Shared Resonance.",
        "Unite your team around the core work. Clear experiment histories, shared context. Build together with genuine understanding. Reproducibility, naturally.",
      )}
    </div>
  </div>
</section>
