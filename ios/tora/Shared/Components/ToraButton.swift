import SwiftUI

// MARK: - Simple, focused API
struct ToraButton: View {
    enum Size { case small, medium, large }

    private let title: String
    private let action: () -> Void
    private let size: Size
    private let backgroundColor: Color
    private let borderColor: Color?
    private let textColor: Color
    private let cornerRadius: CGFloat
    private let fullWidth: Bool
    private let isLoading: Bool
    private let loadingTitle: String?
    private let systemImage: String?

    init(
        _ title: String,
        size: Size = .medium,
        backgroundColor: Color,
        borderColor: Color? = nil,
        textColor: Color = .white,
        cornerRadius: CGFloat = 8,
        fullWidth: Bool = false,
        isLoading: Bool = false,
        loadingTitle: String? = nil,
        systemImage: String? = nil,
        action: @escaping () -> Void
    ) {
        self.title = title
        self.action = action
        self.size = size
        self.backgroundColor = backgroundColor
        self.borderColor = borderColor
        self.textColor = textColor
        self.cornerRadius = cornerRadius
        self.fullWidth = fullWidth
        self.isLoading = isLoading
        self.loadingTitle = loadingTitle
        self.systemImage = systemImage
    }

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView().progressViewStyle(.circular)
                    Text(loadingTitle ?? title)
                } else {
                    if let systemImage { Image(systemName: systemImage) }
                    Text(title)
                }
            }
            .font(font(for: size))
            .frame(maxWidth: fullWidth ? .infinity : nil)
        }
        .buttonStyle(
            ToraButtonStyle(
                size: size,
                backgroundColor: backgroundColor,
                borderColor: borderColor,
                textColor: textColor,
                cornerRadius: cornerRadius
            )
        )
        .disabled(isLoading)
    }

    private func font(for size: Size) -> Font {
        switch size {
        case .small: return .system(.caption, design: .rounded)
        case .medium: return .system(.body, design: .rounded)
        case .large: return .system(.headline, design: .rounded)
        }
    }
}

// MARK: - Minimal style that does just what you asked
struct ToraButtonStyle: ButtonStyle {
    let size: ToraButton.Size
    let backgroundColor: Color
    let borderColor: Color?
    let textColor: Color
    let cornerRadius: CGFloat

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundColor(textColor.opacity(configuration.isPressed ? 0.9 : 1))
            .padding(padding(for: size))
            .background(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .fill(backgroundColor.opacity(configuration.isPressed ? 0.9 : 1))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius)
                    .stroke(borderColor ?? .clear, lineWidth: borderColor == nil ? 0 : 1)
            )
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .animation(.easeInOut(duration: 0.12), value: configuration.isPressed)
    }

    private func padding(for size: ToraButton.Size) -> EdgeInsets {
        switch size {
        case .small: return EdgeInsets(top: 6, leading: 10, bottom: 6, trailing: 10)
        case .medium: return EdgeInsets(top: 10, leading: 14, bottom: 10, trailing: 14)
        case .large: return EdgeInsets(top: 14, leading: 18, bottom: 14, trailing: 18)
        }
    }
}
