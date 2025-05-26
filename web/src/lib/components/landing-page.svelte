<script lang="ts">
  import { UserRound } from "lucide-svelte";
</script>

<section class="flex flex-col h-full p-8 mx-auto text-ctp-text">
  <div class="grid grid-cols-3 grid-rows-2 w-full flex-auto">
    <!-- 1) Top-left: (2/3) -->
    <div class="col-span-2 flex items-end justify-start text-start">
      <h1 class="font-extrabold leading-none text-[clamp(8rem,18vw,32rem)]">TORA</h1>
    </div>

    <!-- 2) Top-right: empty cell (1/3) -->
    <div></div>

    <!-- 3) Bottom-left: empty cell (2/3) -->
    <div class="col-span-2"></div>

    <!-- 4) Bottom-right: (1/3) -->
    <div class="flex flex-col">
      <p class="text-base leading-relaxed">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Dolor sit amet
        consectetur adipiscing elit quisque faucibus. Lorem ipsum dolor sit
        amet, consectetur adipiscing elit. Dolor sit amet consectetur adipiscing
        elit quisque faucibus.
      </p>
      <button class="mt-4 text-start">
        <span class="p-2 rounded-lg bg-ctp-blue/20 hover:bg-ctp-blue transition-colors"
          >Create Experiment</span
        >
      </button>
    </div>
  </div>
  <div class="flex-1">
    <div class="flex flex-row space-x-8">
      <div>
        Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet
        consectetur adipiscing elit quisque faucibus.
      </div>

      <div>
        Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet
        consectetur adipiscing elit quisque faucibus.
      </div>

      <div>
        Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet
        consectetur adipiscing elit quisque faucibus.
      </div>

      <div>
        Lorem ipsum dolor sit amet consectetur adipiscing elit. Dolor sit amet
        consectetur adipiscing elit quisque faucibus.
      </div>
    </div>
  </div>
</section>
