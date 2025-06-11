<script lang="ts">
  let { children } = $props();
</script>

<div
  class="toolbar"
  {@attach (el) => {
    const buttons = el.querySelectorAll(":scope > button");
    for (let i = 0; i < buttons.length; i++) {
      buttons[i].classList.add("toolbar-button");
    }
  }}
>
  {@render children()}
</div>

<style>
  .toolbar {
    position: fixed;
    bottom: 1rem;
    left: 50vw;
    transform: translateX(-50%) scale(1.25);
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 0;
    background-color: var(--color-ctp-surface1);
    border: 1px solid var(--color-ctp-surface2);
    border-radius: 0.5rem;
    box-shadow:
      0 4px 6px -1px rgba(0, 0, 0, 0.1),
      0 2px 4px -1px rgba(0, 0, 0, 0.06);
    z-index: 40;
    overflow: hidden;
    opacity: 100%;
    transition:
      transform 0.3s ease,
      opacity 0.3s ease;
  }

  :global(.toolbar-button) {
    padding: 0.5rem;
    color: var(--color-ctp-subtext0);
    transition:
      color 0.2s,
      background-color 0.2s;
  }

  :global(.toolbar-button:hover) {
    color: var(--color-ctp-text);
    background-color: var(--color-ctp-surface2);
  }

  :global(.icon) {
    width: 20px;
    height: 20px;
  }

  /* Responsive styles */
  @media (min-width: 640px) {
    .toolbar {
      bottom: 3rem;
      gap: 0;
      transform: translateX(-50%) scale(1.1);
    }

    :global(.toolbar-button) {
      padding: 0.625rem;
    }

    :global(.toolbar-button svg) {
      width: 20px;
      height: 20px;
    }
  }

  @media (min-width: 768px) {
    .toolbar {
      width: auto;
      max-width: calc(100vw - 4rem);
      transform: translateX(-50%) scale(1.1);
      opacity: 80%;
    }

    .toolbar:hover {
      transform: translateX(-50%) scale(1.2);
      opacity: 100%;
    }
  }

  @media (min-width: 1024px) {
    .toolbar {
      transform: translateX(-50%) scale(1.2);
      opacity: 80%;
    }

    .toolbar:hover {
      transform: translateX(-50%) scale(1.3);
      opacity: 100%;
    }
  }

  .toolbar.hidden-at-bottom {
    opacity: 0;
    transform: translateX(-50%) translateY(100%) scale(1.25);
    pointer-events: none;
  }

  @media (min-width: 640px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.1);
    }
  }

  @media (min-width: 768px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.1);
    }
  }

  @media (min-width: 1024px) {
    .toolbar.hidden-at-bottom {
      transform: translateX(-50%) translateY(100%) scale(1.2);
    }
  }
</style>
