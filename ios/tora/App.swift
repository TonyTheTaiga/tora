import SwiftData
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var authService: AuthService

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

@main
struct Tora: App {
    @StateObject private var authService = AuthService.shared
    @StateObject private var workspaceService = WorkspaceService(authService: AuthService.shared)

    var body: some Scene {
        WindowGroup {
            ContentView()
                .modalBackground()
                .modelContainer(for: UserSession.self)
                .environmentObject(authService)
                .environmentObject(workspaceService)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AuthService.shared)
}
