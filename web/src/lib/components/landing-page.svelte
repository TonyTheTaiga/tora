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

<!-- Desktop Section -->
<section class="hidden md:flex flex-col h-full w-full text-ctp-text/90">
  <div class="flex-none w-full p-2">
    <span class="font-extrabold leading-none text-[clamp(8rem,20vw,32rem)]">
      TORA
    </span>
  </div>

  <div class="flex justify-end w-full p-2">
    <div class="w-1/3 flex flex-col">
      <p class="text-base leading-relaxed">
        Experiment tracking simplified. Add just 3 lines of Python and
        supercharge your workflow with sleek visualizations, smart AI-driven
        insights, and frictionless collaboration. Built for teams who prefer
        results over complexity.
      </p>
      <button
        class="pt-8 text-start"
        type="button"
        onclick={() => goto("/signup")}
      >
        <span
          class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
          >Start Tracking →</span
        >
      </button>
    </div>
  </div>

  <div class="flex-1 py-4">
    <div class="flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        "Instant Integration",
        "Transform your ML scripts in seconds. Just 3 lines of code for instant compatibility with PyTorch, TensorFlow, scikit-learn, and more.",
      )}
      {@render FeatureCard(
        "Dynamic Visuals",
        "Experience real-time, interactive charts. Effortlessly track, compare, and analyze experiments in a beautifully intuitive dashboard.",
      )}
      {@render FeatureCard(
        "Smart AI Guidance",
        "Leverage AI-driven recommendations to optimize hyperparameters and boost model performance automatically.",
      )}
      {@render FeatureCard(
        "Unified Collaboration",
        "Collaborate seamlessly with shared workspaces, detailed experiment comparisons, and secure role-based access control.",
      )}
    </div>
  </div>
</section>

<!-- Mobile Section -->
<section class="md:hidden flex flex-col h-full w-full text-ctp-text/90 px-4">
  <!-- Title and description -->
  <div class="flex flex-col space-y-8 pt-4">
    <h1 class="font-extrabold leading-none text-[clamp(6rem,30vw,14rem)]">
      TORA
    </h1>
    <p class="text-base leading-relaxed">
      Experiment tracking simplified. Add just 3 lines of Python and supercharge
      your workflow with sleek visualizations, smart AI-driven insights, and
      frictionless collaboration. Built for teams who prefer results over
      complexity.
    </p>
    <button type="button" onclick={() => goto("/signup")}>
      <span
        class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
        >Start Tracking →</span
      >
    </button>
  </div>

  <!-- Mobile Features -->
  <div class="flex-1 py-4">
    <div class="flex flex-col space-y-6">
      {@render FeatureCard(
        "Instant Integration",
        "Transform your ML scripts in seconds. Just 3 lines of code for instant compatibility with PyTorch, TensorFlow, scikit-learn, and more.",
      )}
      {@render FeatureCard(
        "Dynamic Visuals",
        "Experience real-time, interactive charts. Effortlessly track, compare, and analyze experiments in a beautifully intuitive dashboard.",
      )}
      {@render FeatureCard(
        "Smart AI Guidance",
        "Leverage AI-driven recommendations to optimize hyperparameters and boost model performance automatically.",
      )}
      {@render FeatureCard(
        "Unified Collaboration",
        "Collaborate seamlessly with shared workspaces, detailed experiment comparisons, and secure role-based access control.",
      )}
    </div>
  </div>
</section>
