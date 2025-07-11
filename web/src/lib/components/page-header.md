# PageHeader Component

A reusable header component that provides consistent styling and layout for page headers across the application.

## Features

- Consistent visual design with blue accent bar
- Flexible content areas for title, subtitle, description, and additional info
- Optional clickable additional info with custom click handler
- Support for custom action buttons via snippets
- Responsive design with proper truncation

## Props

| Prop                    | Type         | Required | Description                                |
| ----------------------- | ------------ | -------- | ------------------------------------------ |
| `title`                 | `string`     | Yes      | The main page title                        |
| `subtitle`              | `string`     | No       | Secondary text displayed below title       |
| `description`           | `string`     | No       | Additional descriptive text                |
| `additionalInfo`        | `string`     | No       | Extra information (e.g., IDs, counts)      |
| `onAdditionalInfoClick` | `() => void` | No       | Click handler for additional info          |
| `additionalInfoTitle`   | `string`     | No       | Tooltip text for clickable additional info |
| `actionButton`          | `Snippet`    | No       | Custom action button content               |

## Usage Examples

### Basic Header

```svelte
<PageHeader title="Settings" subtitle="system configuration" />
```

### Header with Action Button

```svelte
<PageHeader title="Workspaces" subtitle="5 workspaces">
  {#snippet actionButton()}
    <button onclick={createWorkspace}>New Workspace</button>
  {/snippet}
</PageHeader>
```

### Header with Clickable Additional Info

```svelte
<PageHeader
  title="My Workspace"
  description="A sample workspace"
  subtitle="3 experiments"
  additionalInfo="ws-123456"
  onAdditionalInfoClick={() => copyToClipboard("ws-123456")}
  additionalInfoTitle="Click to copy workspace ID"
>
  {#snippet actionButton()}
    <button onclick={createExperiment}>New Experiment</button>
  {/snippet}
</PageHeader>
```

## Migration

This component replaces the inline header implementations in:

- `/routes/settings/+page.svelte`
- `/routes/workspaces/+page.svelte`
- `/routes/workspaces/[slug]/+page.svelte`
- `/routes/experiments/[experimentId]/+page.svelte` (replaced `ExperimentHeader` component)

The migration maintains all existing functionality while providing a consistent, reusable interface.

## Styling

The component uses the existing Catppuccin color scheme and maintains consistency with the application's design system:

- Blue accent bar (`bg-ctp-blue`)
- Proper text hierarchy with `text-ctp-text` and `text-ctp-subtext0`
- Responsive padding and spacing
- Hover states for interactive elements
