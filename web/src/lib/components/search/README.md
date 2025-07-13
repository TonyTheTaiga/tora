# Search Components

Reusable search components for consistent search functionality across the application.

## Components

### SearchInput

A terminal-style search input with keyboard shortcuts.

**Features:**

- Terminal-style design with optional "/" prefix
- Configurable keyboard shortcuts (default: Ctrl+F)
- Consistent styling with the app theme

**Usage:**

```svelte
<script>
  import { SearchInput } from "$lib/components";

  let searchQuery = $state("");
</script>

<SearchInput bind:value={searchQuery} placeholder="search workspaces..." />
```

**Props:**

- `value` - Bindable search query string
- `placeholder` - Input placeholder text
- `id` - HTML id attribute
- `class` - Additional CSS classes
- `maxWidth` - Container max width (default: "max-w-lg")
- `enableKeyboardShortcuts` - Enable Ctrl+F shortcut (default: true)
- `showSlashPrefix` - Show "/" prefix (default: true)
- `focusKey` - Key for focus shortcut (default: "f")
- `focusKeyRequiresCtrl` - Require Ctrl modifier (default: true)
- `enableEscapeToBlur` - Enable Escape to exit search mode (default: true)
- `onInput` - Callback for input changes

### SearchDropdown

A dropdown search component with filtering and multi-select.

**Features:**

- Dropdown with search filtering
- Multi-select with checkboxes
- Select all/clear all controls
- Keyboard navigation (Ctrl+F to focus, Escape to close)
- Click outside to close

**Usage:**

```svelte
<script>
  import { SearchDropdown } from "$lib/components";

  let metrics = ["accuracy", "loss", "f1_score"];
  let selectedMetrics = $state([]);
  let searchQuery = $state("");
</script>

<SearchDropdown
  items={metrics}
  bind:selectedItems={selectedMetrics}
  bind:searchQuery
  getItemText={(item) => item}
  itemTypeName="metrics"
/>
```

**Props:**

- `items` - Array of items to display
- `selectedItems` - Bindable array of selected items
- `searchQuery` - Bindable search query string
- `placeholder` - Search input placeholder
- `summaryText` - Custom summary text (optional, overrides default)
- `itemTypeName` - Name for item type in summary (default: "items")
- `showControlButtons` - Show all/clear buttons (default: true)
- `enableKeyboardShortcuts` - Enable keyboard shortcuts (default: true)
- `enableEscapeToBlur` - Enable Escape to exit search mode (default: true)
- `getItemText` - Function to get display text for items
- `getItemKey` - Function to get unique key for items (optional)
- `onSelectionChange` - Callback for selection changes
- `onSearchChange` - Callback for search changes

## Keyboard Shortcuts

- **Ctrl+F** - Focus search input
- **Escape** - Exit search mode (blur input) or close dropdown

## Migration Examples

### From workspaces page:

```svelte
<!-- Before -->
<div class="px-4 md:px-6 py-4">
  <div class="max-w-lg">
    <div
      class="flex items-center bg-ctp-surface0/20 focus-within:ring-1 focus-within:ring-ctp-text/20 transition-all"
    >
      <span class="text-ctp-subtext0 font-mono text-sm px-4 py-3">/</span>
      <input
        type="search"
        placeholder="search workspaces..."
        bind:value={searchQuery}
        class="flex-1 bg-transparent border-0 py-3 pr-4 text-ctp-text placeholder-ctp-subtext0 focus:outline-none font-mono text-base"
      />
    </div>
  </div>
</div>

<!-- After -->
<SearchInput bind:value={searchQuery} placeholder="search workspaces..." />
```

### From interactive-chart:

```svelte
<!-- Before -->
<details class="relative" bind:this={detailsElement}>
  <summary
    class="flex items-center justify-between cursor-pointer p-2 md:p-3 bg-ctp-surface0/20 border border-ctp-surface0/30 hover:bg-ctp-surface0/30 transition-colors text-sm md:text-sm"
  >
    <span class="text-ctp-text">
      select metrics ({selectedMetrics.length}/{availableMetrics.length})
    </span>
    <ChevronDown size={12} class="text-ctp-subtext0 md:w-4 md:h-4" />
  </summary>
  <!-- ... complex dropdown content ... -->
</details>

<!-- After -->
<SearchDropdown
  items={availableMetrics}
  bind:selectedItems={selectedMetrics}
  bind:searchQuery={searchFilter}
  getItemText={(metric) => metric}
  itemTypeName="metrics"
/>
```
