import SwiftUI

// MARK: - Reusable UI Components
// Terminal-style components optimized for iOS

// MARK: - Terminal Card Component
public struct TerminalCard<Content: View>: View {
    let content: Content
    let style: CardStyle

    public enum CardStyle {
        case standard
        case elevated
        case accent
        case alternating(isEven: Bool)

        var backgroundColor: Color {
            switch self {
            case .standard:
                return DesignSystem.Colors.surface
            case .elevated:
                return DesignSystem.Colors.surfaceElevated
            case .accent:
                return DesignSystem.Colors.accent.opacity(0.05)
            case .alternating(let isEven):
                return isEven ? DesignSystem.Colors.surface.opacity(0.3) : DesignSystem.Colors.surface.opacity(0.6)
            }
        }

        var borderColor: Color {
            switch self {
            case .standard, .elevated, .alternating:
                return DesignSystem.Colors.border
            case .accent:
                return DesignSystem.Colors.accent.opacity(0.2)
            }
        }
    }

    public init(style: CardStyle = .standard, @ViewBuilder content: () -> Content) {
        self.style = style
        self.content = content()
    }

    public var body: some View {
        content
            .padding(DesignSystem.Spacing.cardPadding)
            .background(style.backgroundColor)
            .overlay(
                Rectangle()
                    .stroke(style.borderColor, lineWidth: 1)
            )
            .cornerRadius(DesignSystem.CornerRadius.card)
    }
}

// MARK: - Terminal Button Component
public struct TerminalButton: View {
    let title: String
    let style: TerminalButtonStyle
    let action: () -> Void

    @State private var isPressed = false

    public init(_ title: String, style: TerminalButtonStyle = .primary, action: @escaping () -> Void) {
        self.title = title
        self.style = style
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(DesignSystem.Typography.callout())
                .foregroundColor(style.foregroundColor)
                .padding(.horizontal, DesignSystem.Spacing.md)
                .padding(.vertical, DesignSystem.Spacing.sm)
                .background(style.backgroundColor)
                .overlay(
                    Rectangle()
                        .stroke(style.borderColor, lineWidth: 1)
                )
                .cornerRadius(DesignSystem.CornerRadius.button)
                .scaleEffect(isPressed ? 0.97 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .onLongPressGesture(
            minimumDuration: 0, maximumDistance: .infinity,
            pressing: { pressing in
                withAnimation(DesignSystem.Animation.buttonPress) {
                    isPressed = pressing
                }
            }, perform: {})
    }
}

// MARK: - Terminal List Item Component
public struct TerminalListItem<Content: View, Actions: View>: View {
    let content: Content
    let actions: Actions?
    let isAlternating: Bool
    let index: Int

    public init(
        index: Int = 0,
        isAlternating: Bool = false,
        @ViewBuilder content: () -> Content,
        @ViewBuilder actions: () -> Actions
    ) {
        self.index = index
        self.isAlternating = isAlternating
        self.content = content()
        self.actions = actions()
    }

    public init(
        index: Int = 0,
        isAlternating: Bool = false,
        @ViewBuilder content: () -> Content
    ) where Actions == EmptyView {
        self.index = index
        self.isAlternating = isAlternating
        self.content = content()
        self.actions = nil
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Main content
            content
                .padding(DesignSystem.Spacing.listItemPadding)

            // Actions section if provided
            if let actions = actions {
                Divider()
                    .background(DesignSystem.Colors.divider)

                HStack {
                    actions
                }
                .padding(.horizontal, DesignSystem.Spacing.listItemPadding)
                .padding(.vertical, DesignSystem.Spacing.sm)
            }
        }
        .background(
            isAlternating
                ? (index % 2 == 0 ? DesignSystem.Colors.surface.opacity(0.3) : DesignSystem.Colors.surface.opacity(0.6))
                : DesignSystem.Colors.surface
        )
        .overlay(
            Rectangle()
                .stroke(DesignSystem.Colors.borderSubtle, lineWidth: 1)
        )
    }
}

// MARK: - Terminal Search Input
public struct TerminalSearchInput: View {
    @Binding var text: String
    let placeholder: String
    let showSlashPrefix: Bool

    @FocusState private var isFocused: Bool

    public init(text: Binding<String>, placeholder: String = "search...", showSlashPrefix: Bool = true) {
        self._text = text
        self.placeholder = placeholder
        self.showSlashPrefix = showSlashPrefix
    }

    public var body: some View {
        HStack(spacing: 0) {
            if showSlashPrefix {
                Text("/")
                    .font(DesignSystem.Typography.mono())
                    .foregroundColor(DesignSystem.Colors.textTertiary)
                    .padding(.leading, DesignSystem.Spacing.md)
            }

            TextField(placeholder, text: $text)
                .font(DesignSystem.Typography.mono())
                .foregroundColor(DesignSystem.Colors.textPrimary)
                .padding(.horizontal, showSlashPrefix ? DesignSystem.Spacing.sm : DesignSystem.Spacing.md)
                .padding(.vertical, DesignSystem.Spacing.sm)
                .focused($isFocused)
        }
        .background(DesignSystem.Colors.surface)
        .overlay(
            Rectangle()
                .stroke(
                    isFocused ? DesignSystem.Colors.accent.opacity(0.5) : DesignSystem.Colors.border,
                    lineWidth: 1
                )
        )
        .cornerRadius(DesignSystem.CornerRadius.input)
        .animation(DesignSystem.Animation.quick, value: isFocused)
    }
}

// MARK: - Terminal Modal Background
public struct TerminalModalBackground<Content: View>: View {
    let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        ZStack {
            // Backdrop
            DesignSystem.Colors.backgroundSecondary
                .opacity(0.9)
                .ignoresSafeArea()
                .background(.ultraThinMaterial)

            // Content
            content
                .padding(DesignSystem.Spacing.screenHorizontal)
        }
    }
}

// MARK: - Terminal Header Component
public struct TerminalHeader: View {
    let title: String
    let subtitle: String?
    let showBackButton: Bool
    let backAction: (() -> Void)?

    public init(
        title: String,
        subtitle: String? = nil,
        showBackButton: Bool = false,
        backAction: (() -> Void)? = nil
    ) {
        self.title = title
        self.subtitle = subtitle
        self.showBackButton = showBackButton
        self.backAction = backAction
    }

    public var body: some View {
        HStack {
            if showBackButton {
                Button(action: backAction ?? {}) {
                    Image(systemName: "chevron.left")
                        .font(.title2)
                        .foregroundColor(DesignSystem.Colors.accent)
                }
                .padding(.trailing, DesignSystem.Spacing.sm)
            }

            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: DesignSystem.Spacing.sm) {
                    Rectangle()
                        .fill(DesignSystem.Colors.accent)
                        .frame(width: 3, height: 20)

                    Text(title)
                        .font(DesignSystem.Typography.title2())
                        .foregroundColor(DesignSystem.Colors.textPrimary)
                }

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(DesignSystem.Typography.caption())
                        .foregroundColor(DesignSystem.Colors.textTertiary)
                        .padding(.leading, DesignSystem.Spacing.md + 3)
                }
            }

            Spacer()
        }
        .padding(.horizontal, DesignSystem.Spacing.screenHorizontal)
        .padding(.vertical, DesignSystem.Spacing.md)
        .background(DesignSystem.Colors.backgroundSecondary)
    }
}

// MARK: - Terminal Empty State
public struct TerminalEmptyState: View {
    let title: String
    let message: String
    let systemImage: String
    let actionTitle: String?
    let action: (() -> Void)?

    public init(
        title: String,
        message: String,
        systemImage: String = "tray",
        actionTitle: String? = nil,
        action: (() -> Void)? = nil
    ) {
        self.title = title
        self.message = message
        self.systemImage = systemImage
        self.actionTitle = actionTitle
        self.action = action
    }

    public var body: some View {
        VStack(spacing: DesignSystem.Spacing.lg) {
            Image(systemName: systemImage)
                .font(.system(size: 48))
                .foregroundColor(DesignSystem.Colors.textTertiary)

            VStack(spacing: DesignSystem.Spacing.sm) {
                Text(title)
                    .font(DesignSystem.Typography.headline())
                    .foregroundColor(DesignSystem.Colors.textPrimary)

                Text(message)
                    .font(DesignSystem.Typography.body())
                    .foregroundColor(DesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
            }

            if let actionTitle = actionTitle, let action = action {
                TerminalButton(actionTitle, style: .primary, action: action)
            }
        }
        .padding(DesignSystem.Spacing.xl)
    }
}

// MARK: - Terminal Loading State
public struct TerminalLoadingState: View {
    let message: String

    public init(message: String = "loading...") {
        self.message = message
    }

    public var body: some View {
        VStack(spacing: DesignSystem.Spacing.md) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle(tint: DesignSystem.Colors.accent))
                .scaleEffect(1.2)

            Text(message)
                .font(DesignSystem.Typography.mono())
                .foregroundColor(DesignSystem.Colors.textSecondary)
        }
        .padding(DesignSystem.Spacing.xl)
    }
}
