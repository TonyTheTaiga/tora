# Modal Components

This directory contains shared modal components that provide consistent styling and behavior across the application.

## Components

### BaseModal

The base modal wrapper that provides consistent styling, backdrop, and structure.

**Props:**

- `title: string` - The modal title displayed in the header

**Usage:**

```svelte
<BaseModal title="My Modal">
  {#snippet children()}
    <!-- Modal content goes here -->
  {/snippet}
</BaseModal>
```

### ModalFormSection

A bordered section container for grouping form elements with a title.

**Props:**

- `title: string` - The section title

**Usage:**

```svelte
<ModalFormSection title="Basic Config">
  {#snippet children()}
    <!-- Form inputs go here -->
  {/snippet}
</ModalFormSection>
```

### ModalInput

A standardized input component with consistent styling.

**Props:**

- `name: string` - Input name attribute
- `type?: string` - Input type (default: "text", use "textarea" for textarea)
- `placeholder?: string` - Placeholder text
- `value?: string` - Bindable value
- `required?: boolean` - Whether the field is required
- `rows?: number` - Number of rows (for textarea type)
- `id?: string` - Input ID attribute

**Usage:**

```svelte
<!-- Text input -->
<ModalInput
  name="title"
  placeholder="Enter title"
  bind:value={title}
  required
/>

<!-- Textarea -->
<ModalInput
  name="description"
  type="textarea"
  rows={3}
  placeholder="Enter description"
  bind:value={description}
/>
```

### ModalButtons

Standardized cancel and submit buttons for modal forms.

**Props:**

- `onCancel: () => void` - Cancel button click handler
- `cancelText?: string` - Cancel button text (default: "cancel")
- `submitText?: string` - Submit button text (default: "submit")
- `isSubmitting?: boolean` - Whether form is submitting (disables submit button)

**Usage:**

```svelte
<ModalButtons onCancel={closeModal} submitText="create" />
```

## Streamlined Imports

You can import all modal components from a single location:

```svelte
<script lang="ts">
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";
</script>
```

Or import specific modals:

```svelte
<script lang="ts">
  import {
    CreateExperimentModal,
    EditExperimentModal,
    DeleteConfirmationModal,
    CreateWorkspaceModal,
    WorkspaceInviteModal,
    DeleteWorkspaceModal,
  } from "$lib/components/modals";
</script>
```

## Example Modal

```svelte
<script lang="ts">
  import {
    BaseModal,
    ModalFormSection,
    ModalInput,
    ModalButtons,
  } from "$lib/components/modals";

  let name = $state("");
  let description = $state("");

  function handleCancel() {
    // Close modal logic
  }
</script>

<BaseModal title="Create Item">
  {#snippet children()}
    <form class="space-y-4">
      <ModalFormSection title="item config">
        {#snippet children()}
          <div>
            <ModalInput
              name="name"
              placeholder="item_name"
              bind:value={name}
              required
            />
          </div>
          <div>
            <ModalInput
              name="description"
              type="textarea"
              rows={2}
              placeholder="description"
              bind:value={description}
            />
          </div>
        {/snippet}
      </ModalFormSection>

      <ModalButtons onCancel={handleCancel} submitText="create" />
    </form>
  {/snippet}
</BaseModal>
```

## Benefits

1. **Consistent Styling**: All modals use the same visual design and spacing
2. **Reduced Code Duplication**: Shared components eliminate repeated styling code
3. **Maintainability**: Changes to modal styling only need to be made in one place
4. **Type Safety**: Components provide proper TypeScript types
5. **Accessibility**: Built-in ARIA attributes and proper semantic structure
6. **Responsive**: Consistent responsive behavior across all modals
7. **Streamlined Imports**: Single import location for all modal components
8. **Better Developer Experience**: Clean, organized imports reduce cognitive load
