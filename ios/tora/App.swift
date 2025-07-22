import SwiftData
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
                .background(Color.white)
                .modelContainer(for: UserSession.self)
                .environmentObject(authService)
                .environmentObject(workspaceService)
                .environmentObject(experimentService)
        }
    }
}

// MARK: - Preview

#Preview {
    ContentView()
        .environmentObject(AuthService.shared)
        .environmentObject(WorkspaceService(authService: AuthService.shared))
        .environmentObject(ExperimentService(authService: AuthService.shared))
}
