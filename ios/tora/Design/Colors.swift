import SwiftUI

extension Color {
    // Dynamic Colors
    static var background: Color {
        .init(light: Color(white: 1.0), dark: Color(white: 0.05))
    }
    static var secondaryBackground: Color {
        .init(light: Color(white: 0.95), dark: Color(white: 0.1))
    }
    static var tertiaryBackground: Color {
        .init(light: Color(white: 0.9), dark: Color(white: 0.15))
    }
    static var textPrimary: Color {
        .init(light: Color(white: 0.1), dark: Color(white: 0.9))
    }
    static var textSecondary: Color {
        .init(light: Color(white: 0.4), dark: Color(white: 0.6))
    }
    static var accent: Color {
        .init(light: Color.blue, dark: Color.blue)
    }
    static var accent2: Color {
        .init(light: Color.indigo, dark: Color.indigo)
    }
}

extension Color {
    init(light: Color, dark: Color) {
        self = UIColor(light: UIColor(light), dark: UIColor(dark)).toColor()
    }
}

extension UIColor {
    convenience init(light: UIColor, dark: UIColor) {
        self.init { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .light, .unspecified:
                return light
            case .dark:
                return dark
            @unknown default:
                return light
            }
        }
    }

    func toColor() -> Color {
        Color(self)
    }
}
