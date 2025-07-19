import SwiftUI

// MARK: - Design Tokens
// Comprehensive design token system inspired by web's Catppuccin implementation
// Provides semantic color mapping and theme support

public struct DesignTokens {

    // MARK: - Theme Management
    public enum Theme: String, CaseIterable {
        case light
        case dark
        case system

        public var displayName: String {
            switch self {
            case .light: return "Light"
            case .dark: return "Dark"
            case .system: return "System"
            }
        }
    }

    // MARK: - Semantic Color Tokens
    public struct SemanticColors {
        // Background hierarchy
        public static var backgroundPrimary: Color { Color.ctpBase }
        public static var backgroundSecondary: Color { Color.ctpMantle }
        public static var backgroundTertiary: Color { Color.ctpCrust }

        // Surface layers (inspired by web's surface system)
        public static var surfaceBase: Color { Color.ctpSurface0 }
        public static var surfaceElevated: Color { Color.ctpSurface1 }
        public static var surfaceHighest: Color { Color.ctpSurface2 }

        // Interactive surfaces
        public static var surfaceInteractive: Color { Color.ctpSurface0.opacity(0.8) }
        public static var surfaceHover: Color { Color.ctpSurface0.opacity(0.5) }
        public static var surfacePressed: Color { Color.ctpSurface1.opacity(0.8) }

        // Text hierarchy
        public static var textPrimary: Color { Color.ctpText }
        public static var textSecondary: Color { Color.ctpSubtext1 }
        public static var textTertiary: Color { Color.ctpSubtext0 }
        public static var textDisabled: Color { Color.ctpOverlay0 }

        // Brand colors (inspired by web's accent system)
        public static var brandPrimary: Color { Color.ctpBlue }
        public static var brandSecondary: Color { Color.ctpLavender }
        public static var brandTertiary: Color { Color.ctpMauve }

        // Status colors
        public static var statusSuccess: Color { Color.ctpGreen }
        public static var statusWarning: Color { Color.ctpYellow }
        public static var statusError: Color { Color.ctpRed }
        public static var statusInfo: Color { Color.ctpSapphire }

        // Border and divider system
        public static var borderPrimary: Color { Color.ctpSurface0.opacity(0.3) }
        public static var borderSecondary: Color { Color.ctpSurface0.opacity(0.15) }
        public static var borderAccent: Color { Color.ctpBlue.opacity(0.3) }
        public static var borderFocus: Color { Color.ctpBlue.opacity(0.5) }

        // Overlay colors (for modals, sheets, etc.)
        public static var overlayLight: Color { Color.ctpBase.opacity(0.8) }
        public static var overlayMedium: Color { Color.ctpBase.opacity(0.9) }
        public static var overlayHeavy: Color { Color.ctpBase.opacity(0.95) }
    }

    // MARK: - Button Color Tokens
    public struct ButtonColors {
        public static var primaryBackground: Color { SemanticColors.brandPrimary.opacity(0.1) }
        public static var primaryForeground: Color { SemanticColors.brandPrimary }
        public static var primaryBorder: Color { SemanticColors.brandPrimary.opacity(0.3) }

        public static var secondaryBackground: Color { SemanticColors.surfaceBase }
        public static var secondaryForeground: Color { SemanticColors.textPrimary }
        public static var secondaryBorder: Color { SemanticColors.borderPrimary }

        public static var destructiveBackground: Color { SemanticColors.statusError.opacity(0.1) }
        public static var destructiveForeground: Color { SemanticColors.statusError }
        public static var destructiveBorder: Color { SemanticColors.statusError.opacity(0.3) }
    }

    // MARK: - Input Color Tokens
    public struct InputColors {
        public static var background: Color { SemanticColors.surfaceBase }
        public static var foreground: Color { SemanticColors.textPrimary }
        public static var placeholder: Color { SemanticColors.textTertiary }
        public static var border: Color { SemanticColors.borderPrimary }
        public static var borderFocused: Color { SemanticColors.borderFocus }
    }

    // MARK: - Card Color Tokens
    public struct CardColors {
        public static var background: Color { SemanticColors.surfaceBase }
        public static var backgroundElevated: Color { SemanticColors.surfaceElevated }
        public static var backgroundAccent: Color { SemanticColors.brandPrimary.opacity(0.05) }
        public static var border: Color { SemanticColors.borderPrimary }
        public static var borderAccent: Color { SemanticColors.borderAccent }
    }

    // MARK: - List Color Tokens
    public struct ListColors {
        public static var backgroundEven: Color { SemanticColors.surfaceBase.opacity(0.3) }
        public static var backgroundOdd: Color { SemanticColors.surfaceBase.opacity(0.6) }
        public static var backgroundHover: Color { SemanticColors.surfaceHover }
        public static var border: Color { SemanticColors.borderSecondary }
    }

    // MARK: - Navigation Color Tokens
    public struct NavigationColors {
        public static var background: Color { SemanticColors.backgroundSecondary }
        public static var backgroundGlass: Color { SemanticColors.surfaceBase.opacity(0.8) }
        public static var foreground: Color { SemanticColors.textPrimary }
        public static var accent: Color { SemanticColors.brandPrimary }
    }

    // MARK: - Typography Tokens
    public struct TypographyTokens {
        // Font sizes (responsive)
        public static let fontSizeXS: CGFloat = 11
        public static let fontSizeSM: CGFloat = 12
        public static let fontSizeMD: CGFloat = 14
        public static let fontSizeLG: CGFloat = 16
        public static let fontSizeXL: CGFloat = 18
        public static let fontSize2XL: CGFloat = 20
        public static let fontSize3XL: CGFloat = 24
        public static let fontSize4XL: CGFloat = 28
        public static let fontSize5XL: CGFloat = 32

        // Line heights
        public static let lineHeightTight: CGFloat = 1.2
        public static let lineHeightNormal: CGFloat = 1.5
        public static let lineHeightRelaxed: CGFloat = 1.75

        // Font weights
        public static let fontWeightRegular: Font.Weight = .regular
        public static let fontWeightMedium: Font.Weight = .medium
        public static let fontWeightSemibold: Font.Weight = .semibold
        public static let fontWeightBold: Font.Weight = .bold
    }

    // MARK: - Spacing Tokens
    public struct SpacingTokens {
        // Base spacing scale
        public static let space0: CGFloat = 0
        public static let space1: CGFloat = 2
        public static let space2: CGFloat = 4
        public static let space3: CGFloat = 8
        public static let space4: CGFloat = 12
        public static let space5: CGFloat = 16
        public static let space6: CGFloat = 20
        public static let space7: CGFloat = 24
        public static let space8: CGFloat = 32
        public static let space9: CGFloat = 40
        public static let space10: CGFloat = 48
        public static let space11: CGFloat = 56
        public static let space12: CGFloat = 64

        // Semantic spacing
        public static let spacingXS: CGFloat = space2
        public static let spacingSM: CGFloat = space3
        public static let spacingMD: CGFloat = space5
        public static let spacingLG: CGFloat = space7
        public static let spacingXL: CGFloat = space8
        public static let spacing2XL: CGFloat = space10
        public static let spacing3XL: CGFloat = space12

        // Component spacing
        public static let componentPaddingXS: CGFloat = space2
        public static let componentPaddingSM: CGFloat = space3
        public static let componentPaddingMD: CGFloat = space5
        public static let componentPaddingLG: CGFloat = space7

        // Layout spacing
        public static let layoutMarginXS: CGFloat = space3
        public static let layoutMarginSM: CGFloat = space5
        public static let layoutMarginMD: CGFloat = space7
        public static let layoutMarginLG: CGFloat = space8
    }

    // MARK: - Border Radius Tokens
    public struct RadiusTokens {
        public static let radiusNone: CGFloat = 0
        public static let radiusXS: CGFloat = 2
        public static let radiusSM: CGFloat = 4
        public static let radiusMD: CGFloat = 8
        public static let radiusLG: CGFloat = 12
        public static let radiusXL: CGFloat = 16
        public static let radiusFull: CGFloat = 9999

        // Component-specific radii (inspired by web's sharp terminal style)
        public static let cardRadius: CGFloat = radiusNone  // Sharp edges
        public static let buttonRadius: CGFloat = radiusMD  // Rounded for navigation
        public static let inputRadius: CGFloat = radiusNone  // Sharp for terminal style
        public static let modalRadius: CGFloat = radiusNone  // Sharp for consistency
    }

    // MARK: - Shadow Tokens
    public struct ShadowTokens {
        public static let shadowNone = Color.clear
        public static let shadowXS = Color.black.opacity(0.03)
        public static let shadowSM = Color.black.opacity(0.05)
        public static let shadowMD = Color.black.opacity(0.1)
        public static let shadowLG = Color.black.opacity(0.15)
        public static let shadowXL = Color.black.opacity(0.2)

    }

    // MARK: - Shadow Configuration Tokens
    public struct ShadowConfig {
        public let color: Color
        public let radius: CGFloat
        public let offset: CGSize

        public static let subtle = ShadowConfig(
            color: ShadowTokens.shadowSM,
            radius: 4,
            offset: CGSize(width: 0, height: 2)
        )

        public static let medium = ShadowConfig(
            color: ShadowTokens.shadowMD,
            radius: 8,
            offset: CGSize(width: 0, height: 4)
        )

        public static let large = ShadowConfig(
            color: ShadowTokens.shadowLG,
            radius: 16,
            offset: CGSize(width: 0, height: 8)
        )
    }

    // MARK: - Animation Tokens
    public struct AnimationTokens {
        // Duration tokens
        public static let durationInstant: Double = 0
        public static let durationFast: Double = 0.1
        public static let durationNormal: Double = 0.2
        public static let durationSlow: Double = 0.3
        public static let durationSlower: Double = 0.4
        public static let durationSlowest: Double = 0.6

        // Easing curves
        public static let easeLinear = Animation.linear
        public static let easeIn = Animation.easeIn
        public static let easeOut = Animation.easeOut
        public static let easeInOut = Animation.easeInOut

        // Semantic animations
        public static let animationQuick = Animation.easeInOut(duration: durationFast)
        public static let animationSmooth = Animation.easeInOut(duration: durationNormal)
        public static let animationGentle = Animation.easeInOut(duration: durationSlow)
        public static let animationDelayed = Animation.easeInOut(duration: durationSlower)

        // Interaction animations
        public static let buttonPress = Animation.easeInOut(duration: durationFast)
        public static let modalPresent = Animation.easeInOut(duration: durationSlow)
        public static let listUpdate = Animation.easeInOut(duration: durationNormal)
        public static let pageTransition = Animation.easeInOut(duration: durationSlower)
    }

    // MARK: - Breakpoint Tokens
    public struct BreakpointTokens {
        public static let extraSmall: CGFloat = 320  // Small phones
        public static let small: CGFloat = 375  // Regular phones
        public static let medium: CGFloat = 414  // Large phones
        public static let large: CGFloat = 768  // Small tablets
        public static let extraLarge: CGFloat = 1024  // Large tablets
        public static let xxl: CGFloat = 1200  // Desktop
    }
}
