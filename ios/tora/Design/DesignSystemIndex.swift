import SwiftUI

// MARK: - Design System Index
// Central export file for the entire design system
// Provides easy access to all design system components and tokens

// MARK: - Design System Interface
// Simple interface for accessing design system components
// Use the individual types directly for better type safety

// MARK: - Responsive Typography Helper
public struct ResponsiveTypography {
    public static func title1() -> Font {
        DesignSystem.Typography.responsiveTitle1()
    }

    public static func title2() -> Font {
        DesignSystem.Typography.responsiveTitle2()
    }

    public static func body() -> Font {
        DesignSystem.Typography.responsiveBody()
    }

    public static func caption() -> Font {
        DesignSystem.Typography.responsiveCaption()
    }
}

// MARK: - Design System Utilities
extension View {
    // Quick access to common design system modifiers
    public func toraCard(style: TerminalCard<AnyView>.CardStyle = .standard) -> some View {
        TerminalCard(style: style) {
            AnyView(self)
        }
    }

    public func toraContainer(maxWidth: CGFloat? = nil) -> some View {
        TerminalContainer(maxWidth: maxWidth) {
            AnyView(self)
        }
    }

    public func toraResponsivePadding() -> some View {
        self.responsivePadding()
    }

    public func toraTerminalAccent(color: Color = DesignTokens.SemanticColors.brandPrimary) -> some View {
        self.terminalAccent(color: color)
    }

    public func toraAnimatedAppearance(delay: Double = 0) -> some View {
        self.animatedAppearance(delay: delay)
    }
}

// MARK: - Design System Constants
public struct ToraDesignConstants {
    // Common measurements
    public static let minTouchTarget: CGFloat = 44
    public static let standardLineHeight: CGFloat = 1.5
    public static let goldenRatio: CGFloat = 1.618

    // Layout constants
    public static let maxContentWidth: CGFloat = 800
    public static let sidebarWidth: CGFloat = 280
    public static let navigationHeight: CGFloat = 56

    // Animation constants
    public static let defaultAnimationDuration: Double = 0.3
    public static let quickAnimationDuration: Double = 0.15
    public static let slowAnimationDuration: Double = 0.5

    // Accessibility constants
    public static let minimumContrastRatio: Double = 4.5
    public static let preferredContrastRatio: Double = 7.0
}

// MARK: - Design System Validation
#if DEBUG
    public struct DesignSystemValidator {
        // Validation helpers for ensuring design system compliance
        public static func validateColorContrast(_ foreground: Color, _ background: Color) -> Bool {
            // Implementation would check WCAG contrast ratios
            return true  // Placeholder
        }

        public static func validateTouchTarget(_ size: CGSize) -> Bool {
            return size.width >= ToraDesignConstants.minTouchTarget && size.height >= ToraDesignConstants.minTouchTarget
        }

        public static func validateSpacing(_ spacing: CGFloat) -> Bool {
            let validSpacings = [
                DesignTokens.SpacingTokens.space0,
                DesignTokens.SpacingTokens.space1,
                DesignTokens.SpacingTokens.space2,
                DesignTokens.SpacingTokens.space3,
                DesignTokens.SpacingTokens.space4,
                DesignTokens.SpacingTokens.space5,
                DesignTokens.SpacingTokens.space6,
                DesignTokens.SpacingTokens.space7,
                DesignTokens.SpacingTokens.space8,
                DesignTokens.SpacingTokens.space9,
                DesignTokens.SpacingTokens.space10,
                DesignTokens.SpacingTokens.space11,
                DesignTokens.SpacingTokens.space12,
            ]
            return validSpacings.contains(spacing)
        }
    }
#endif

// MARK: - Design System Documentation
public struct DesignSystemDocs {
    public static let version = "1.0.0"
    public static let description = """
        Tora iOS Design System

        A comprehensive design system inspired by the web terminal-style approach,
        optimized for iOS with mobile-specific patterns and responsive design.

        Key Features:
        - Catppuccin color palette with light/dark theme support
        - Inter typography with dynamic scaling
        - Terminal-style, information-first component design
        - Responsive layout system for iPhone and iPad
        - Comprehensive design tokens
        - Accessibility-first approach

        Usage:
        Import the design system and use ToraDesignSystem as the main interface
        for accessing colors, typography, spacing, and components.
        """

    public static let principles = [
        "Information-first: Data and content take priority over decoration",
        "Terminal aesthetic: Sharp edges, monospace elements, technical feel",
        "Responsive design: Adapts seamlessly from iPhone to iPad",
        "Accessibility: WCAG compliant with proper contrast and touch targets",
        "Consistency: Unified visual language across all components",
        "Performance: Optimized for smooth animations and interactions",
    ]

    public static let colorPalette = """
        Color System based on Catppuccin:
        - Base colors: Background hierarchy (base, mantle, crust)
        - Surface colors: Interactive element backgrounds (surface0, surface1, surface2)
        - Text colors: Typography hierarchy (text, subtext1, subtext0)
        - Brand colors: Primary interactions (blue, lavender, mauve)
        - Status colors: Feedback and states (green, yellow, red, sapphire)
        """
}

// MARK: - Export All Design System Components
// This ensures all components are available when importing the design system

// Re-export core design system
public typealias ToraColors = DesignSystem.Colors
public typealias ToraTypography = DesignSystem.Typography
public typealias ToraSpacing = DesignSystem.Spacing
public typealias ToraCornerRadius = DesignSystem.CornerRadius
public typealias ToraShadow = DesignSystem.Shadow
public typealias ToraAnimation = DesignSystem.Animation

// Re-export components
public typealias ToraCard = TerminalCard
public typealias ToraButton = TerminalButton
public typealias ToraListItem = TerminalListItem
public typealias ToraSearchInput = TerminalSearchInput
public typealias ToraHeader = TerminalHeader
public typealias ToraEmptyState = TerminalEmptyState
public typealias ToraLoadingState = TerminalLoadingState
public typealias ToraModalBackground = TerminalModalBackground

// Re-export layout components
public typealias ToraContainer = TerminalContainer
public typealias ToraScreen = TerminalScreen
public typealias ToraList = TerminalList
public typealias ToraGrid = TerminalGrid
public typealias ToraAdaptiveStack = AdaptiveStack
public typealias ToraSafeAreaContainer = SafeAreaContainer

// Re-export design tokens
public typealias ToraDesignTokens = DesignTokens
public typealias ToraSemanticColors = DesignTokens.SemanticColors
public typealias ToraComponentColors = DesignTokens.ComponentColors
public typealias ToraTypographyTokens = DesignTokens.TypographyTokens
public typealias ToraSpacingTokens = DesignTokens.SpacingTokens
public typealias ToraRadiusTokens = DesignTokens.RadiusTokens
public typealias ToraShadowTokens = DesignTokens.ShadowTokens
public typealias ToraAnimationTokens = DesignTokens.AnimationTokens
public typealias ToraBreakpointTokens = DesignTokens.BreakpointTokens
