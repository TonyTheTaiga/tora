import SwiftUI

// MARK: - Content View

struct ContentView: View {
    // MARK: - Properties

    @EnvironmentObject var authService: AuthService

    // MARK: - Body

    var body: some View {
        Group {
            if authService.isAuthenticated {
                MainAppView()
            } else {
                LoginView()
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
                .task {
                    do {
                        if let user = authService.currentUser, user.expiresIn < Date() {
                            try await authService.refreshUserSession()
                        }
                    } catch {
                        // TODO: fix this, maybe manage the state inside auth service so that the caller never has to handle it.
                        print("Failed to refresh session: \(error.localizedDescription)")
                    }
                }
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
