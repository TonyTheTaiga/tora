# Tora iOS Design System

A comprehensive design system inspired by the web terminal-style approach, optimized for iOS with mobile-specific patterns and responsive design.

## Overview

The Tora iOS Design System provides a unified visual language that maintains consistency with the web application while being optimized for mobile interactions and iOS design patterns.

### Key Features

- **Catppuccin Color Palette**: Full light/dark theme support with semantic color mapping
- **Inter Typography**: Dynamic scaling with accessibility support
- **Terminal Aesthetic**: Sharp edges, information-first design, technical feel
- **Responsive Design**: Seamless adaptation from iPhone to iPad
- **Component Library**: Pre-built, reusable UI components
- **Design Tokens**: Comprehensive token system for consistency
- **Accessibility First**: WCAG compliant with proper contrast ratios

## Architecture

```
Design/
├── Colors.swift              # Catppuccin color definitions (existing)
├── Typography.swift          # Inter font system (existing)
├── Icons.swift              # Icon components (existing)
├── DesignSystem.swift       # Core design system foundation
├── DesignTokens.swift       # Comprehensive design tokens
├── Components.swift         # Reusable UI components
├── Layout.swift            # Layout system and responsive helpers
├── DesignSystemIndex.swift # Central export interface
└── README.md              # This documentation
```

## Quick Start

### Basic Usage

```swift
import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack(spacing: ToraSpacing.md) {
            // Using design system components
            ToraHeader(title: "Experiments", subtitle: "12 total")

            ToraCard(style: .standard) {
                Text("Terminal-style card")
                    .font(ToraTypography.body())
                    .foregroundColor(ToraColors.textPrimary)
            }

            ToraButton("Create Experiment", style: .primary) {
                // Action
            }
        }
        .toraContainer()
        .background(ToraColors.background)
    }
}
```

### Using Design Tokens

```swift
// Colors
.foregroundColor(ToraDesignTokens.SemanticColors.textPrimary)
.background(ToraDesignTokens.ComponentColors.Card.background)

// Typography
.font(ToraDesignTokens.TypographyTokens.fontSizeLG)

// Spacing
.padding(ToraDesignTokens.SpacingTokens.spacingMD)

// Animations
.animation(ToraDesignTokens.AnimationTokens.animationSmooth)
```

## Design Principles

### 1. Information-First
Data and content take priority over decorative elements. The design emphasizes clarity and readability.

### 2. Terminal Aesthetic
- Sharp edges (no rounded corners on cards/containers)
- Monospace fonts for data display
- Technical, developer-focused visual language
- Subtle borders and dividers

### 3. Responsive Design
- Mobile-first approach
- Dynamic typography scaling
- Adaptive layouts for different screen sizes
- Touch-optimized interactions

### 4. Accessibility
- WCAG 2.1 AA compliance
- Minimum 4.5:1 contrast ratios
- Dynamic type support
- Proper touch target sizes (44pt minimum)

## Color System

### Semantic Colors
```swift
// Background hierarchy
ToraColors.background          // Primary background
ToraColors.backgroundSecondary // Secondary background
ToraColors.backgroundTertiary  // Tertiary background

// Text hierarchy
ToraColors.textPrimary    // Primary text
ToraColors.textSecondary  // Secondary text
ToraColors.textTertiary   // Tertiary text

// Interactive elements
ToraColors.accent         // Primary brand color (blue)
ToraColors.accentSecondary // Secondary brand color (lavender)
ToraColors.accentTertiary  // Tertiary brand color (mauve)
```

### Component Colors
```swift
// Button colors
ToraComponentColors.Button.primaryBackground
ToraComponentColors.Button.primaryForeground
ToraComponentColors.Button.primaryBorder

// Card colors
ToraComponentColors.Card.background
ToraComponentColors.Card.backgroundElevated
ToraComponentColors.Card.border
```

## Typography System

### Responsive Typography
```swift
// Automatically scales with device size and accessibility settings
.font(ToraTypography.responsiveTitle1())  // 28pt on iPhone, 32pt on iPad
.font(ToraTypography.responsiveBody())    // 16pt on compact, 17pt on regular
.font(ToraTypography.responsiveCaption()) // 11pt on compact, 12pt on regular
```

### Standard Typography
```swift
.font(ToraTypography.title1())     // 28pt, bold
.font(ToraTypography.headline())   // 17pt, semibold
.font(ToraTypography.body())       // 17pt, regular
.font(ToraTypography.caption())    // 12pt, regular
.font(ToraTypography.mono())       // Monospace for data
```

## Component Library

### Cards
```swift
// Standard card
ToraCard(style: .standard) {
    Text("Content")
}

// Elevated card
ToraCard(style: .elevated) {
    Text("Content")
}

// Alternating list cards
ToraCard(style: .alternating(isEven: index % 2 == 0)) {
    Text("Content")
}
```

### Buttons
```swift
// Primary button
ToraButton("Submit", style: .primary) { }

// Secondary button
ToraButton("Cancel", style: .secondary) { }

// Destructive button
ToraButton("Delete", style: .destructive) { }
```

### Lists
```swift
ToraList(experiments) { experiment, index in
    ToraListItem(index: index, isAlternating: true) {
        // Content
        Text(experiment.name)
    } actions: {
        // Action buttons
        ToraButton("Edit", style: .secondary) { }
    }
}
```

### Search Input
```swift
ToraSearchInput(
    text: $searchQuery,
    placeholder: "search experiments...",
    showSlashPrefix: true
)
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
