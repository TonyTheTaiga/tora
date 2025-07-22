import SwiftData
import SwiftUI

struct MainAppView: View {
    // MARK: - Properties

    @State private var selectedTab: Tabs = .workspaces
    @State private var selectedExperimentId: String?

    // MARK: - Tabs

    enum Tabs: Equatable, Hashable {
        case workspaces
        case experiments
        case settings
    }

    // MARK: - Body

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab("Workspaces", systemImage: "macwindow.stack", value: .workspaces) {
                NavigationStack {
                    WorkspacesView(onExperimentSelected: { experimentId in
                        selectedExperimentId = experimentId
                        selectedTab = .experiments
                    })
                }
            }

            Tab("Experiments", systemImage: "flask", value: .experiments) {
                NavigationStack {
                    ExperimentsView(experimentId: selectedExperimentId)
                }
            }

            Tab("Settings", systemImage: "gearshape.2", value: .settings) {
                SettingsView()
            }
        }.accentColor(.blue)
    }
}

// MARK: - Preview

#Preview {
    MainAppView()
        .environmentObject(AuthService.shared)
        .environmentObject(WorkspaceService(authService: AuthService.shared))
        .environmentObject(ExperimentService(authService: AuthService.shared))
        .modelContainer(for: UserSession.self, inMemory: true)
        .onAppear {
            let service = AuthService.shared
            service.isAuthenticated = true
            service.currentUser = UserSession(
                id: "preview-user",
                email: "preview@tora.com",
                authToken: "token",
                refreshToken: "refresh",
                expiresIn: Date(),
                expiresAt: Date().addingTimeInterval(3600),
                tokenType: "Bearer"
            )
        }
}
