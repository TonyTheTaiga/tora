import SwiftUI

private struct AppThemeKey: EnvironmentKey {
    static let defaultValue: AppTheme = .system
}

extension EnvironmentValues {
    var appTheme: AppTheme {
        get { self[AppThemeKey.self] }
        set { self[AppThemeKey.self] = newValue }
    }
}

private struct AppThemeModifier: ViewModifier {
    @AppStorage("appTheme") private var selectedThemeRaw: String = AppTheme.system.rawValue

    private var selectedTheme: AppTheme {
        AppTheme(rawValue: selectedThemeRaw) ?? .system
    }

    func body(content: Content) -> some View {
        content
            .environment(\.appTheme, selectedTheme)
            .preferredColorScheme(selectedTheme.colorScheme)
    }
}

extension View {
    func applyAppTheme() -> some View { modifier(AppThemeModifier()) }
}
