import { browser } from "$app/environment";

export function createClipboardState() {
  let copied = $state(false);
  let timeoutId: ReturnType<typeof setTimeout> | undefined;

  function copyToClipboard(text: string) {
    if (!browser) return;

    navigator.clipboard.writeText(text).then(() => {
      copied = true;

      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = setTimeout(() => {
        copied = false;
      }, 2000);
    });
  }

  function getCopied() {
    return copied;
  }

  return {
    copyToClipboard,
    getCopied,
  };
}

export function createScrollState() {
  let isAtBottom = $state(false);

  function updateScrollState() {
    if (!browser) return;

    const pageIsScrollable =
      document.documentElement.scrollHeight > window.innerHeight;
    const atActualBottom =
      window.innerHeight + Math.ceil(window.scrollY) >=
      document.documentElement.scrollHeight - 1;

    if (pageIsScrollable && atActualBottom) {
      isAtBottom = true;
    } else {
      isAtBottom = false;
    }
  }

  function getIsAtBottom() {
    return isAtBottom;
  }

  return {
    updateScrollState,
    getIsAtBottom,
  };
}

export function createLoadingState() {
  let isLoading = $state(false);
  let loadingMessage = $state<string | null>(null);

  function setLoading(loading: boolean, message?: string) {
    isLoading = loading;
    loadingMessage = message || null;
  }

  function getIsLoading() {
    return isLoading;
  }

  function getLoadingMessage() {
    return loadingMessage;
  }

  return {
    setLoading,
    getIsLoading,
    getLoadingMessage,
  };
}

export function createToggleState(initialValue = false) {
  let value = $state(initialValue);

  function toggle() {
    value = !value;
  }

  function setValue(newValue: boolean) {
    value = newValue;
  }

  function getValue() {
    return value;
  }

  return {
    toggle,
    setValue,
    getValue,
  };
}
