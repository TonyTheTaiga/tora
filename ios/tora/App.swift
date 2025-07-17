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

    var body: some Scene {
        WindowGroup {
            ContentView()
                .modalBackground()
                .preferredColorScheme(.light)
                .modelContainer(for: UserSession.self)
                .environmentObject(authService)
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(AuthService.shared)
}
