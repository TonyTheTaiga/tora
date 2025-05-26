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

<section class="flex flex-col h-full w-full text-ctp-text/90">
  <!-- Desktop Layout -->
  <div class="hidden md:flex flex-col w-full p-2">
    <!-- Top section with TORA title -->
    <div class="flex items-end text-end">
      <span class="font-extrabold leading-none text-[clamp(8rem,20vw,32rem)]">
        TORA
      </span>
    </div>

    <!-- Bottom section with description -->
    <div class="flex justify-end">
      <div class="w-1/3 flex flex-col">
        <p class="text-base leading-relaxed">
          The ML experiment tracking platform that gets out of your way. Add
          just 3 lines to your Python code and unlock powerful experiment
          management, interactive visualizations, and AI-powered insights.
          Perfect for ML teams who want enterprise capabilities without the
          complexity.
        </p>
        <button
          class="pt-8 text-start"
          type="button"
          onclick={() => {
            goto("/signup");
          }}
        >
          <span
            class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
            >Get Started -></span
          >
        </button>
      </div>
    </div>
  </div>

  <!-- Mobile Layout -->
  <div class="md:hidden flex flex-col space-y-8 flex-auto px-4">
    <!-- Title -->
    <div>
      <h1 class="font-extrabold leading-none text-[clamp(6rem,30vw,14rem)]">
        TORA
      </h1>
    </div>

    <!-- Description and CTA -->
    <div class="flex flex-col space-y-4">
      <p class="text-base leading-relaxed text-left">
        The ML experiment tracking platform that gets out of your way. Add just
        3 lines to your Python code and unlock powerful experiment management,
        interactive visualizations, and AI-powered insights. Perfect for ML
        teams who want enterprise capabilities without the complexity.
      </p>
      <div>
        <button
          type="button"
          onclick={() => {
            goto("/signup");
          }}
        >
          <span
            class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
            >Get Started -></span
          >
        </button>
      </div>
    </div>
  </div>

  <!-- Features Section -->
  <div class="flex-1 py-4">
    <!-- Desktop: horizontal layout -->
    <div class="hidden md:flex flex-row space-x-8 p-2">
      {@render FeatureCard(
        "Zero-Friction Integration",
        "Add experiment tracking to any ML script with just 3 lines of code. Works with PyTorch, TensorFlow, scikit-learn, and any Python ML framework.",
      )}
      {@render FeatureCard(
        "Interactive Visualizations",
        "Beautiful charts and real-time metric tracking. Compare experiments side-by-side with powerful filtering and analysis tools.",
      )}
      {@render FeatureCard(
        "AI-Powered Insights",
        "Get intelligent hyperparameter recommendations and optimization suggestions to improve your model performance automatically.",
      )}
      {@render FeatureCard(
        "Team Collaboration",
        "Shared workspaces, experiment comparison across team members, and role-based access control for seamless ML team collaboration.",
      )}
    </div>

    <!-- Mobile: vertical layout -->
    <div class="md:hidden flex flex-col space-y-6 px-4">
      {@render FeatureCard(
        "Zero-Friction Integration",
        "Add experiment tracking to any ML script with just 3 lines of code. Works with PyTorch, TensorFlow, scikit-learn, and any Python ML framework.",
      )}
      {@render FeatureCard(
        "Interactive Visualizations",
        "Beautiful charts and real-time metric tracking. Compare experiments side-by-side with powerful filtering and analysis tools.",
      )}
      {@render FeatureCard(
        "AI-Powered Insights",
        "Get intelligent hyperparameter recommendations and optimization suggestions to improve your model performance automatically.",
      )}
      {@render FeatureCard(
        "Team Collaboration",
        "Shared workspaces, experiment comparison across team members, and role-based access control for seamless ML team collaboration.",
      )}
    </div>
  </div>
</section>
