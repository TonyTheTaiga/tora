import SwiftUI

// MARK: - Reusable UI Components
// iOS-native components with Tora design language

// MARK: - Tora Card Component
public struct ToraCard<Content: View>: View {
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
                return .clear
            case .elevated:
                return Color.secondary.opacity(0.1)
            case .accent:
                return Color.blue.opacity(0.1)
            case .alternating(let isEven):
                return isEven ? Color.secondary.opacity(0.05) : Color.secondary.opacity(0.1)
            }
        }

        var borderColor: Color {
            switch self {
            case .standard, .elevated, .alternating:
                return Color.secondary.opacity(0.3)
            case .accent:
                return Color.blue.opacity(0.3)
            }
        }
    }

    public init(style: CardStyle = .standard, @ViewBuilder content: () -> Content) {
        self.style = style
        self.content = content()
    }

    public var body: some View {
        content
            .padding(16)
            .background(style.backgroundColor, in: RoundedRectangle(cornerRadius: 12))
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(style.borderColor, lineWidth: 0.5)
            )
    }
}

// MARK: - Tora Button Component
public struct ToraButton: View {
    let title: String
    let style: ToraButtonStyle
    let action: () -> Void

    @State private var isPressed = false

    public init(_ title: String, style: ToraButtonStyle = .primary, action: @escaping () -> Void) {
        self.title = title
        self.style = style
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Text(title)
                .font(.callout)
                .fontWeight(.medium)
                .foregroundColor(style.foregroundColor)
                .padding(.horizontal, 16)
                .padding(.vertical, 10)
                .background(style.backgroundColor, in: RoundedRectangle(cornerRadius: 8))
                .scaleEffect(isPressed ? 0.97 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .onLongPressGesture(
            minimumDuration: 0, maximumDistance: .infinity,
            pressing: { pressing in
                withAnimation(.easeInOut(duration: 0.1)) {
                    isPressed = pressing
                }
            }, perform: {})
    }
}

// MARK: - Tora Toolbar Button Component
public struct ToraToolbarButton: View {
    let systemImage: String
    let action: () -> Void

    @State private var isPressed = false

    public init(systemImage: String, action: @escaping () -> Void) {
        self.systemImage = systemImage
        self.action = action
    }

    public var body: some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .font(.body)
                .foregroundColor(.accentColor)
                .scaleEffect(isPressed ? 0.95 : 1.0)
        }
        .buttonStyle(PlainButtonStyle())
        .onLongPressGesture(
            minimumDuration: 0, maximumDistance: .infinity,
            pressing: { pressing in
                withAnimation(.easeInOut(duration: 0.1)) {
                    isPressed = pressing
                }
            }, perform: {})
    }
}

// MARK: - Tora List Item Component
public struct ToraListItem<Content: View, Actions: View>: View {
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
                .padding(12)

            // Actions section if provided
            if let actions = actions {
                Divider()

                HStack {
                    actions
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
        }
        .background(
            isAlternating
                ? (index % 2 == 0 ? Color.secondary.opacity(0.05) : Color.secondary.opacity(0.1))
                : Color.clear,
            in: RoundedRectangle(cornerRadius: 8)
        )
    }
}

// MARK: - Tora Search Input
public struct ToraSearchInput: View {
    @Binding var text: String
    let placeholder: String

    @FocusState private var isFocused: Bool

    public init(text: Binding<String>, placeholder: String = "Search...") {
        self._text = text
        self.placeholder = placeholder
    }

    public var body: some View {
        HStack {
            Image(systemName: "magnifyingglass")
                .foregroundColor(.secondary)

            TextField(placeholder, text: $text)
                .focused($isFocused)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(isFocused ? Color(.systemBlue) : Color.clear, lineWidth: 2)
        )
        .animation(.easeInOut(duration: 0.2), value: isFocused)
    }
}

// MARK: - Tora Modal Background
public struct ToraModalBackground<Content: View>: View {
    let content: Content

    public init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    public var body: some View {
        ZStack {
            Color.black.opacity(0.3)
                .ignoresSafeArea()

            content
                .padding(20)
        }
    }
}

// MARK: - Tora Header Component
public struct ToraHeader: View {
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
                        .foregroundColor(.accentColor)
                }
                .padding(.trailing, 8)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .foregroundColor(.primary)

                if let subtitle = subtitle {
                    Text(subtitle)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }

            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 16)
    }
}

// MARK: - Tora Empty State
public struct ToraEmptyState: View {
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
        VStack(spacing: 24) {
            Image(systemName: systemImage)
                .font(.system(size: 48))
                .foregroundColor(.secondary)

            VStack(spacing: 8) {
                Text(title)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text(message)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }

            if let actionTitle = actionTitle, let action = action {
                ToraButton(actionTitle, style: .primary, action: action)
            }
        }
        .padding(32)
    }
}

// MARK: - Tora Loading State
public struct ToraLoadingState: View {
    let message: String

    public init(message: String = "loading...") {
        self.message = message
    }

    public var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.2)

            Text(message)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(32)
    }
}
