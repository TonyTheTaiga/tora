import SwiftUI

// MARK: - Content View

struct ContentView: View {
    // MARK: - Properties

    @EnvironmentObject var authService: AuthService

    // MARK: - Body

    var body: some View {
        Group {
            switch authService.state {
            case .authenticated(_):
                MainAppView()
            case .unauthenticated(let message):
                if let message, !message.isEmpty {
                    VStack(spacing: 12) {
                        Text(message).foregroundStyle(.red)
                        LoginView()
                    }
                } else {
                    LoginView()
                }
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
                    if let userSession = authService.state.userSession,
                        userSession.expiresIn < Date().addingTimeInterval(60)
                    {
                        await authService.refreshUserSession()
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
