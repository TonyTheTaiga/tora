import SwiftData
import SwiftUI

struct MainAppView: View {
    @State private var selectedTab: Tabs = .workspaces

    enum Tabs: Equatable, Hashable {
        case workspaces
        case experiments
        case settings
    }

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Workspaces", systemImage: "macwindow.stack", value: .workspaces) {
                WorkspacesView()
            }

            //            Tab("Experiments", systemImage: "sparkle.text.clipboard", value: .experiments) {
            //                ExperimentsView()
            //            }

            Tab("Settings", systemImage: "gearshape.2", value: .settings) {
                ExperimentsView()
            }
        }.accentColor(Color.ctpOverlay2)
    }
}

#Preview {
    MainAppView()
        .environmentObject(AuthService.shared)
        .environmentObject(WorkspaceService(authService: AuthService.shared))
        .modelContainer(for: UserSession.self, inMemory: true)
        .onAppear {
            let service = AuthService.shared
            service.isAuthenticated = true
            service.currentUser = UserSession(
                id: "preview-user",
                email: "preview@tora.com",
                auth_token: "token",
                refresh_token: "refresh",
                expiresIn: Date(),
                expiresAt: Date().addingTimeInterval(3600),
                tokenType: "Bearer"
            )
        }
        .modalBackground()
}
