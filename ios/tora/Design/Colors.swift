import SwiftUI

// MARK: - Color Palette

struct ColorPalette {
    let ctpBase = Color(light: "#eff1f5", dark: "#1e1e2e")
    let ctpMantle = Color(light: "#e6e9ef", dark: "#181825")
    let ctpCrust = Color(light: "#dce0e8", dark: "#11111b")
    let ctpSurface0 = Color(light: "#ccd0da", dark: "#313244")
    let ctpSurface1 = Color(light: "#bcc0cc", dark: "#45475a")
    let ctpSurface2 = Color(light: "#acb0be", dark: "#585b70")
    let ctpOverlay0 = Color(light: "#9ca0b0", dark: "#6c7086")
    let ctpOverlay1 = Color(light: "#8c8fa1", dark: "#7f849c")
    let ctpOverlay2 = Color(light: "#7c7f93", dark: "#9399b2")
    let ctpText = Color(light: "#4c4f69", dark: "#e0e7ff")
    let ctpSubtext0 = Color(light: "#6c6f85", dark: "#a6adc8")
    let ctpSubtext1 = Color(light: "#5c5f77", dark: "#bac2de")
    let ctpLavender = Color(light: "#7287fd", dark: "#b4befe")
    let ctpBlue = Color(light: "#1e66f5", dark: "#89b4fa")
    let ctpSapphire = Color(light: "#209fb5", dark: "#74c7ec")
    let ctpSky = Color(light: "#04a5e5", dark: "#89dceb")
    let ctpTeal = Color(light: "#179299", dark: "#94e2d5")
    let ctpGreen = Color(light: "#40a02b", dark: "#a6e3a1")
    let ctpYellow = Color(light: "#df8e1d", dark: "#f9e2af")
    let ctpPeach = Color(light: "#fe640b", dark: "#fab387")
    let ctpMaroon = Color(light: "#e64553", dark: "#eba0ac")
    let ctpRed = Color(light: "#d20f39", dark: "#f38ba8")
    let ctpMauve = Color(light: "#8839ef", dark: "#cba6f7")
    let ctpPink = Color(light: "#ea76cb", dark: "#f5c2e7")
    let ctpFlamingo = Color(light: "#dd7878", dark: "#f2cdcd")
    let ctpRosewater = Color(light: "#dc8a78", dark: "#f5e0dc")
    let ctpSheetBackground = Color(light: "#eff1f5", dark: "#1e1e2e")
}

// MARK: - Color Extension

extension Color {
    static let custom = ColorPalette()
}

extension Color {
    init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a: UInt64
        let r: UInt64
        let g: UInt64
        let b: UInt64
        switch hex.count {
        case 3:  // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:  // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:  // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)  // Default to clear or black
        }

        self.init(
            .sRGB,
            red: Double(r) / 255,
            green: Double(g) / 255,
            blue: Double(b) / 255,
            opacity: Double(a) / 255
        )
    }

    init(light: String, dark: String) {
        self.init(uiColor: UIColor(light: light, dark: dark))
    }
}

// MARK: - UIColor Extension

extension UIColor {
    convenience init(light: String, dark: String) {
        self.init { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor(hex: dark) : UIColor(hex: light)
        }
    }

    convenience init(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: hex).scanHexInt64(&int)
        let a: UInt64
        let r: UInt64
        let g: UInt64
        let b: UInt64
        switch hex.count {
        case 3:  // RGB (12-bit)
            (a, r, g, b) = (255, (int >> 8) * 17, (int >> 4 & 0xF) * 17, (int & 0xF) * 17)
        case 6:  // RGB (24-bit)
            (a, r, g, b) = (255, int >> 16, int >> 8 & 0xFF, int & 0xFF)
        case 8:  // ARGB (32-bit)
            (a, r, g, b) = (int >> 24, int >> 16 & 0xFF, int >> 8 & 0xFF, int & 0xFF)
        default:
            (a, r, g, b) = (1, 1, 1, 0)  // Default to clear or black
        }

        self.init(
            red: CGFloat(r) / 255,
            green: CGFloat(g) / 255,
            blue: CGFloat(b) / 255,
            alpha: CGFloat(a) / 255
        )
    }
}
