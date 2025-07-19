import SwiftUI

// MARK: - Design System Foundation
public struct DesignSystem {

    // MARK: - Color System
    public struct Colors {
        // Base semantic colors that adapt to light/dark mode
        public static var background: Color { Color.ctpBase }
        public static var backgroundSecondary: Color { Color.ctpMantle }
        public static var backgroundTertiary: Color { Color.ctpCrust }

        // Surface layers for depth and hierarchy
        public static var surface: Color { Color.ctpSurface0 }
        public static var surfaceElevated: Color { Color.ctpSurface1 }
        public static var surfaceHighest: Color { Color.ctpSurface2 }

        // Text hierarchy
        public static var textPrimary: Color { Color.ctpText }
        public static var textSecondary: Color { Color.ctpSubtext1 }
        public static var textTertiary: Color { Color.ctpSubtext0 }

        // Interactive elements
        public static var accent: Color { Color.ctpBlue }
        public static var accentSecondary: Color { Color.ctpLavender }
        public static var accentTertiary: Color { Color.ctpMauve }

        // Status colors
        public static var success: Color { Color.ctpGreen }
        public static var warning: Color { Color.ctpYellow }
        public static var error: Color { Color.ctpRed }
        public static var info: Color { Color.ctpSapphire }

        // Border and divider colors
        public static var border: Color { Color.ctpSurface0.opacity(0.3) }
        public static var borderSubtle: Color { Color.ctpSurface0.opacity(0.15) }
        public static var divider: Color { Color.ctpSurface0.opacity(0.2) }
    }

    // MARK: - Typography System
    public struct Typography {
        // Dynamic typography that scales with accessibility settings
        public static func title1() -> Font {
            Font.dynamicInter(28, weight: .bold, relativeTo: .largeTitle)
        }

        public static func title2() -> Font {
            Font.dynamicInter(22, weight: .bold, relativeTo: .title)
        }

        public static func title3() -> Font {
            Font.dynamicInter(20, weight: .semibold, relativeTo: .title2)
        }

        public static func headline() -> Font {
            Font.dynamicInter(17, weight: .semibold, relativeTo: .headline)
        }

        public static func body() -> Font {
            Font.dynamicInter(17, weight: .regular, relativeTo: .body)
        }

        public static func bodyEmphasized() -> Font {
            Font.dynamicInter(17, weight: .medium, relativeTo: .body)
        }

        public static func callout() -> Font {
            Font.dynamicInter(16, weight: .regular, relativeTo: .callout)
        }

        public static func subheadline() -> Font {
            Font.dynamicInter(15, weight: .regular, relativeTo: .subheadline)
        }

        public static func footnote() -> Font {
            Font.dynamicInter(13, weight: .regular, relativeTo: .footnote)
        }

        public static func caption() -> Font {
            Font.dynamicInter(12, weight: .regular, relativeTo: .caption)
        }

        // Terminal-style monospace for data display
        public static func mono() -> Font {
            Font.system(.body, design: .monospaced)
        }

        public static func monoSmall() -> Font {
            Font.system(.caption, design: .monospaced)
        }
    }

    // MARK: - Spacing System
    public struct Spacing {
        public static let extraSmall: CGFloat = 4
        public static let small: CGFloat = 8
        public static let medium: CGFloat = 16
        public static let large: CGFloat = 24
        public static let extraLarge: CGFloat = 32
        public static let xxl: CGFloat = 48

        // Screen margins
        public static let screenHorizontal: CGFloat = 16
        public static let screenVertical: CGFloat = 20

        // Component-specific spacing
        public static let cardPadding: CGFloat = 16
        public static let listItemPadding: CGFloat = 12
        public static let buttonPadding: CGFloat = 12
    }

    // MARK: - Corner Radius System
    public struct CornerRadius {
        public static let none: CGFloat = 0
        public static let small: CGFloat = 4
        public static let medium: CGFloat = 8
        public static let large: CGFloat = 12
        public static let extraLarge: CGFloat = 16

        // Component-specific radii
        public static let card: CGFloat = 0  // Sharp edges for terminal style
        public static let button: CGFloat = 8  // Rounded for navigation elements
        public static let input: CGFloat = 0  // Sharp for terminal style
        public static let modal: CGFloat = 0  // Sharp for consistency
    }

    // MARK: - Shadow System
    public struct Shadow {
        public static let none = Color.clear
        public static let subtle = Color.black.opacity(0.05)
        public static let medium = Color.black.opacity(0.1)
        public static let strong = Color.black.opacity(0.15)

        // Shadow configurations
        public static func cardShadow() -> some View {
            Rectangle()
                .fill(subtle)
                .blur(radius: 4)
                .offset(y: 2)
        }

        public static func elevatedShadow() -> some View {
            Rectangle()
                .fill(medium)
                .blur(radius: 8)
                .offset(y: 4)
        }
    }

    // MARK: - Animation System
    public struct Animation {
        public static let quick = SwiftUI.Animation.easeInOut(duration: 0.2)
        public static let smooth = SwiftUI.Animation.easeInOut(duration: 0.3)
        public static let gentle = SwiftUI.Animation.easeInOut(duration: 0.4)
        public static let slow = SwiftUI.Animation.easeInOut(duration: 0.6)

        // Interactive animations
        public static let buttonPress = SwiftUI.Animation.easeInOut(duration: 0.1)
        public static let modalPresent = SwiftUI.Animation.easeInOut(duration: 0.3)
        public static let listUpdate = SwiftUI.Animation.easeInOut(duration: 0.25)
    }
}

// MARK: - Component Style Extensions
extension View {
    func toraCard() -> some View {
        self
            .padding(16)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    func toraCardElevated() -> some View {
        self
            .padding(16)
            .background(.thickMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    func glassBackground() -> some View {
        self
            .background(.ultraThinMaterial)
    }

    func toraButton(style: ToraButtonStyle = .primary) -> some View {
        self
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(style.backgroundColor, in: RoundedRectangle(cornerRadius: 8))
            .foregroundColor(style.foregroundColor)
    }

    func toraListItem() -> some View {
        self
            .padding(12)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Button Styles
public enum ToraButtonStyle {
    case primary
    case secondary
    case accent
    case destructive

    public var backgroundColor: Color {
        switch self {
        case .primary:
            return Color(.systemBlue)
        case .secondary:
            return Color(.secondarySystemFill)
        case .accent:
            return Color(.systemPurple)
        case .destructive:
            return Color(.systemRed)
        }
    }

    public var foregroundColor: Color {
        switch self {
        case .primary:
            return Color(.white)
        case .secondary:
            return Color(.label)
        case .accent:
            return Color(.white)
        case .destructive:
            return Color(.white)
        }
    }
}
