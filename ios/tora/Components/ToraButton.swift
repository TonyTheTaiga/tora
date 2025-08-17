import SwiftUI

struct ToraButton: View {
    enum Size {
        case small, medium, large
    }

    enum Variant {
        case tinted, filled, outline, plain
    }

    private let title: String
    private let loadingTitle: String?
    private let systemImage: String?
    private let action: () -> Void

    // Configurable properties
    private let size: Size
    private let variant: Variant
    private let fullWidth: Bool
    private let accent: Color
    private let textColor: Color
    private let cornerRadius: CGFloat
    private let isLoading: Bool

    init(
        _ title: String,
        loadingTitle: String? = nil,
        systemImage: String? = nil,
        size: Size = .small,
        variant: Variant = .tinted,
        fullWidth: Bool = false,
        accent: Color = Color.custom.ctpBlue,
        textColor: Color = Color.custom.ctpText,
        cornerRadius: CGFloat = 6,
        isLoading: Bool = false,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.loadingTitle = loadingTitle
        self.systemImage = systemImage
        self.size = size
        self.variant = variant
        self.fullWidth = fullWidth
        self.accent = accent
        self.textColor = textColor
        self.cornerRadius = cornerRadius
        self.isLoading = isLoading
        self.action = action
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                    Text(loadingTitle ?? title)
                } else {
                    if let systemImage = systemImage {
                        Image(systemName: systemImage)
                    }
                    Text(title)
                }
            }
            .font(fontForSize(size))
            .frame(maxWidth: fullWidth ? .infinity : nil)
        }
        .buttonStyle(
            ToraButtonStyle(
                size: size,
                variant: variant,
                fullWidth: fullWidth,
                accent: accent,
                textColor: textColor,
                cornerRadius: cornerRadius
            )
        )
    }

    private func fontForSize(_ size: Size) -> Font {
        switch size {
        case .small: return .system(.caption, design: .monospaced)
        case .medium: return .system(.body, design: .monospaced)
        case .large: return .system(.headline, design: .monospaced)
        }
    }
}

struct ToraButtonStyle: ButtonStyle {
    let size: ToraButton.Size
    let variant: ToraButton.Variant
    let fullWidth: Bool
    let accent: Color
    let textColor: Color
    let cornerRadius: CGFloat

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(foregroundColor(isPressed: configuration.isPressed))
            .padding(paddingForSize(size))
            .background(background(isPressed: configuration.isPressed))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(borderColor(isPressed: configuration.isPressed), lineWidth: borderWidth())
            )
            .cornerRadius(cornerRadius)
            .scaleEffect(configuration.isPressed ? 0.985 : 1.0)
            .shadow(
                color: shadowColor(isPressed: configuration.isPressed),
                radius: 4, x: 0, y: 2
            )
            .animation(.easeInOut(duration: 0.12), value: configuration.isPressed)
    }

    private func paddingForSize(_ size: ToraButton.Size) -> EdgeInsets {
        switch size {
        case .small: return EdgeInsets(top: 6, leading: 10, bottom: 6, trailing: 10)
        case .medium: return EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14)
        case .large: return EdgeInsets(top: 14, leading: 18, bottom: 14, trailing: 18)
        }
    }

    private func borderWidth() -> CGFloat {
        switch variant {
        case .plain: return 0
        default: return 1
        }
    }

    private func foregroundColor(isPressed: Bool) -> Color {
        switch variant {
        case .filled:
            return Color.custom.ctpCrust
        case .tinted, .outline:
            return isPressed ? Color.custom.ctpCrust : textColor
        case .plain:
            return textColor
        }
    }

    private func background(isPressed: Bool) -> some View {
        let base: Color
        switch variant {
        case .filled:
            base = accent.opacity(isPressed ? 0.9 : 1.0)
        case .tinted:
            base = accent.opacity(isPressed ? 0.35 : 0.20)
        case .outline:
            base = Color.clear
        case .plain:
            base = Color.clear
        }
        return RoundedRectangle(cornerRadius: cornerRadius).fill(base)
    }

    private func borderColor(isPressed: Bool) -> Color {
        switch variant {
        case .filled:
            return accent.opacity(isPressed ? 1.0 : 0.9)
        case .tinted:
            return accent.opacity(isPressed ? 0.8 : 0.6)
        case .outline:
            return accent.opacity(isPressed ? 0.9 : 0.6)
        case .plain:
            return .clear
        }
    }

    private func shadowColor(isPressed: Bool) -> Color {
        switch variant {
        case .plain:
            return .clear
        default:
            return accent.opacity(isPressed ? 0.0 : 0.12)
        }
    }
}
