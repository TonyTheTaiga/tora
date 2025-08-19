import SwiftUI

// MARK: - Content View

struct ContentView: View {
    // MARK: - Properties

    @EnvironmentObject var authService: AuthService
    @State private var error: String = ""

    // MARK: - Body

    var body: some View {
        Group {
            if authService.isAuthenticated {
                MainAppView()
            } else if !error.isEmpty {
                Text(error)
            } else {
                LoginView()
            }
        }.task {
            do {
                if let user = authService.currentUser, user.expiresIn < Date() {
                    try await authService.refreshUserSession()
                }
            } catch {
                self.error = "Failed to refresh session: \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Tora App

@main
struct Tora: App {
    // MARK: - Properties

    @StateObject private var authService = AuthService.shared
    @StateObject private var workspaceService = WorkspaceService(authService: AuthService.shared)
    @StateObject private var experimentService = ExperimentService(authService: AuthService.shared)

    // MARK: - Body

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(authService)
                .environmentObject(workspaceService)
                .environmentObject(experimentService)
                .applyAppTheme()
        }
    }
}

// MARK: - Theme Application

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
    fileprivate func applyAppTheme() -> some View { modifier(AppThemeModifier()) }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environmentObject(AuthService.shared)
        .environmentObject(WorkspaceService(authService: AuthService.shared))
        .environmentObject(ExperimentService(authService: AuthService.shared))
}
