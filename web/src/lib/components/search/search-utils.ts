/**
 * Utility functions for search functionality
 */

/**
 * Filter items based on search query
 */
export function filterItems<T>(
  items: T[],
  searchQuery: string,
  getSearchableText: (item: T) => string,
): T[] {
  if (!searchQuery.trim()) return items;

  const query = searchQuery.toLowerCase().trim();
  return items.filter((item) =>
    getSearchableText(item).toLowerCase().includes(query),
  );
}

/**
 * Focus search input by selector
 */
export function focusSearchInput(selector = 'input[type="search"]'): void {
  const searchElement = document.querySelector<HTMLInputElement>(selector);
  searchElement?.focus();
}

/**
 * Blur search input by selector
 */
export function blurSearchInput(selector = 'input[type="search"]'): void {
  const searchElement = document.querySelector<HTMLInputElement>(selector);
  searchElement?.blur();
}

/**
 * Create keyboard event handler for search shortcuts
 */
export function createSearchKeyboardHandler(options: {
  onFocusKey?: () => void;
  onEscapeKey?: () => void;
  searchInputSelector?: string;
  shouldIgnoreWhenInputFocused?: boolean;
  focusKey?: string;
  focusKeyRequiresCtrl?: boolean;
  enableEscapeToBlur?: boolean;
}) {
  const {
    onFocusKey,
    onEscapeKey,
    searchInputSelector = 'input[type="search"]',
    shouldIgnoreWhenInputFocused = true,
    focusKey = "f",
    focusKeyRequiresCtrl = true,
    enableEscapeToBlur = true,
  } = options;

  return (event: KeyboardEvent) => {
    // Handle focus key (default: Ctrl+F)
    if (event.key.toLowerCase() === focusKey.toLowerCase()) {
      // Skip if user is typing in an input/textarea (unless it's the search input)
      if (
        shouldIgnoreWhenInputFocused &&
        (event.target instanceof HTMLInputElement ||
          event.target instanceof HTMLTextAreaElement)
      ) {
        return;
      }

      if (!focusKeyRequiresCtrl || (focusKeyRequiresCtrl && event.ctrlKey)) {
        event.preventDefault();
        if (onFocusKey) {
          onFocusKey();
        } else {
          focusSearchInput(searchInputSelector);
        }
        return;
      }
    }

    // Handle Escape key
    if (event.key === "Escape") {
      // If escape is pressed while in a search input, blur it
      if (
        enableEscapeToBlur &&
        event.target instanceof HTMLInputElement &&
        event.target.type === "search"
      ) {
        event.preventDefault();
        event.target.blur();
        return;
      }

      // Otherwise, call custom escape handler
      if (onEscapeKey) {
        event.preventDefault();
        onEscapeKey();
      }
    }
  };
}

/**
 * Debounce function for search input
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number,
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout>;

  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Toggle item in array (for multi-select)
 */
export function toggleArrayItem<T>(array: T[], item: T): T[] {
  const index = array.indexOf(item);
  if (index === -1) {
    return [...array, item];
  } else {
    return array.filter((_, i) => i !== index);
  }
}

/**
 * Check if item is selected in array
 */
export function isItemSelected<T>(array: T[], item: T): boolean {
  return array.includes(item);
}
