# Tora iOS Design System

A native iOS design system that provides consistent, accessible, and beautiful UI components for the Tora experiment tracking app.

## Overview

The Tora iOS Design System embraces iOS design principles while maintaining the information-first philosophy of the Tora brand. It provides native-feeling components that work seamlessly across iPhone and iPad.

### Key Features

- **iOS Native**: Uses system colors, materials, and design patterns
- **Dynamic Typography**: Supports Dynamic Type and accessibility scaling
- **Information-First**: Clean, data-focused design approach
- **Responsive Design**: Adapts beautifully from iPhone to iPad
- **Component Library**: Pre-built, reusable UI components
- **Accessibility First**: Full VoiceOver and accessibility support

## Architecture

```
Design/
├── Colors.swift         # Catppuccin color definitions (existing)
├── Typography.swift     # Inter font system (existing)
├── Icons.swift         # Icon components (existing)
├── DesignSystem.swift  # Core design system foundation
├── DesignTokens.swift  # Design tokens for consistency
├── Components.swift    # Reusable UI components
├── Layout.swift       # Layout system and responsive helpers
└── README.md          # This documentation
```

## Quick Start

### Basic Usage

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: 16) {
            // Using design system components
            ToraHeader(title: "Experiments", subtitle: "12 total")

            ToraCard(style: .standard) {
                Text("Clean, native iOS card")
                    .font(.body)
                    .foregroundColor(.primary)
            }

            ToraButton("Create Experiment", style: .primary) {
                // Action
            }
        }
        .padding()
        .background(Color(.systemBackground))
    }
}
```

### Using System Colors

```swift
// Native iOS colors that adapt to light/dark mode
.foregroundColor(.primary)
.background(.regularMaterial)
.accentColor(.blue)

// Custom semantic colors
.foregroundColor(Color(.label))
.background(Color(.systemBackground))
```

## Design Principles

### 1. Information-First
Data and content take priority over decorative elements. The design emphasizes clarity and readability while feeling native to iOS.

### 2. iOS Native
- Uses system colors and materials for automatic light/dark mode
- Follows iOS design patterns and conventions
- Leverages platform capabilities like Dynamic Type
- Feels familiar to iOS users

### 3. Responsive Design
- Mobile-first approach optimized for touch
- Dynamic typography that scales with accessibility settings
- Adaptive layouts for iPhone and iPad
- Proper touch target sizes (44pt minimum)

### 4. Accessibility
- Full VoiceOver support
- Dynamic Type support
- High contrast mode support
- Proper semantic labeling

## Color System

### System Colors
```swift
// iOS system colors that automatically adapt to light/dark mode
Color(.systemBackground)      // Primary background
Color(.secondarySystemBackground) // Secondary background
Color(.tertiarySystemBackground)  // Tertiary background

// Text colors
Color(.label)           // Primary text
Color(.secondaryLabel)  // Secondary text
Color(.tertiaryLabel)   // Tertiary text

// Interactive elements
Color(.systemBlue)      // Primary actions
Color(.systemPurple)    // Secondary actions
Color(.systemRed)       // Destructive actions
```

### Materials
```swift
// iOS materials for depth and layering
.regularMaterial        // Standard material
.thinMaterial          // Subtle material
.ultraThinMaterial     // Very subtle material
.thickMaterial         // Strong material
```

## Typography System

### Dynamic Typography
```swift
// iOS system fonts that automatically scale with accessibility settings
.font(.largeTitle)     // Largest title
.font(.title)          // Section titles
.font(.title2)         // Subsection titles
.font(.headline)       // Important content
.font(.body)           // Regular content
.font(.callout)        // Emphasized content
.font(.subheadline)    // Secondary content
.font(.footnote)       // Small content
.font(.caption)        // Smallest content
```

### Font Weights
```swift
.fontWeight(.ultraLight)
.fontWeight(.thin)
.fontWeight(.light)
.fontWeight(.regular)
.fontWeight(.medium)
.fontWeight(.semibold)
.fontWeight(.bold)
.fontWeight(.heavy)
.fontWeight(.black)
```

## Component Library

### Cards
```swift
// Standard card with native iOS styling
ToraCard(style: .standard) {
    Text("Content")
}

// Elevated card with secondary background
ToraCard(style: .elevated) {
    Text("Content")
}

// Alternating cards for lists
ToraCard(style: .alternating(isEven: index % 2 == 0)) {
    Text("Content")
}
```

### Buttons
```swift
// Primary button (system blue)
ToraButton("Submit", style: .primary) { }

// Secondary button (system fill)
ToraButton("Cancel", style: .secondary) { }

// Destructive button (system red)
ToraButton("Delete", style: .destructive) { }
```

### Toolbar Buttons
```swift
// Icon-based toolbar buttons for navigation
ToraToolbarButton(systemImage: "xmark") {
    // Close/cancel action
}

ToraToolbarButton(systemImage: "arrow.clockwise") {
    // Refresh action
}

ToraToolbarButton(systemImage: "plus") {
    // Add action
}
```

### Search Input
```swift
ToraSearchInput(text: $searchQuery, placeholder: "Search...")
```

### Empty States
```swift
ToraEmptyState(
    title: "No Data",
    message: "There's nothing here yet.",
    systemImage: "tray",
    actionTitle: "Add Item"
) {
    // Action
}
```

### Loading States
```swift
ToraLoadingState(message: "Loading...")
```

## Layout System

### Containers
```swift
// Responsive container with proper margins
ToraContainer(maxWidth: 800) {
    content
}

// Screen-level container with scroll support
ToraScreen {
    content
}
```

### Responsive Helpers
```swift
// Adaptive stack (VStack on phone, HStack on iPad)
ToraAdaptiveStack {
    content
}

// Responsive padding
.toraResponsivePadding()

// Conditional modifiers
.if(DeviceInfo.isPad) { view in
    view.padding(.horizontal, 32)
}
```

## Spacing System

### Design Tokens
```swift
ToraSpacingTokens.spacingXS   // 4pt
ToraSpacingTokens.spacingSM   // 8pt
ToraSpacingTokens.spacingMD   // 16pt
ToraSpacingTokens.spacingLG   // 24pt
ToraSpacingTokens.spacingXL   // 32pt
```

### Responsive Spacing
```swift
ResponsiveSpacing.horizontal()    // 16pt on phone, 24pt on iPad
ResponsiveSpacing.vertical()      // 16pt on phone, 32pt on iPad
ResponsiveSpacing.cardPadding()   // 8pt on compact, 16pt on regular
```

## Animation System

### Standard Animations
```swift
.animation(ToraAnimationTokens.animationQuick)   // 0.1s for interactions
.animation(ToraAnimationTokens.animationSmooth)  // 0.2s for transitions
.animation(ToraAnimationTokens.animationGentle)  // 0.3s for presentations
```

### Component Animations
```swift
.animation(ToraAnimationTokens.buttonPress)     // Button interactions
.animation(ToraAnimationTokens.modalPresent)    // Modal presentations
.animation(ToraAnimationTokens.listUpdate)      // List changes
```

## Best Practices

### 1. Use Semantic Colors
Always use semantic color tokens instead of direct Catppuccin colors:
```swift
// ✅ Good
.foregroundColor(ToraColors.textPrimary)

// ❌ Avoid
.foregroundColor(Color.ctpText)
```

### 2. Responsive Design
Consider different screen sizes and orientations:
```swift
// ✅ Good
.font(ToraTypography.responsiveBody())
.padding(ResponsiveSpacing.horizontal())

// ❌ Avoid
.font(.system(size: 17))
.padding(.horizontal, 16)
```

### 3. Accessibility
Ensure proper contrast and touch targets:
```swift
// ✅ Good
ToraButton("Action", style: .primary) { }  // 44pt touch target

// ❌ Avoid
Button("Action") { }
.frame(width: 30, height: 30)  // Too small
```

### 4. Consistent Spacing
Use design tokens for all spacing:
```swift
// ✅ Good
VStack(spacing: ToraSpacingTokens.spacingMD) { }

// ❌ Avoid
VStack(spacing: 15) { }  // Arbitrary value
```

## Migration from Existing Code

To migrate existing views to use the design system:

1. Replace color references with semantic tokens
2. Update typography to use design system fonts
3. Replace custom spacing with design tokens
4. Wrap content in design system containers
5. Use design system components where applicable

## Contributing

When adding new components or tokens:

1. Follow the established naming conventions
2. Ensure accessibility compliance
3. Add responsive behavior where appropriate
4. Document usage examples
5. Update this README with new additions

## Version History

- **1.0.0**: Initial design system implementation
  - Core design tokens
  - Component library
  - Layout system
  - Responsive design helpers
