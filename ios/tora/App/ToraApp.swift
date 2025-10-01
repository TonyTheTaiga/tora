import Combine
import SwiftUI

final class AppServices: ObservableObject {
    let authService: AuthService
    let workspaceService: WorkspaceService
    let experimentService: ExperimentService

    init() {
        let authService = AuthService()
        self.authService = authService
        self.workspaceService = WorkspaceService(authService: authService)
        self.experimentService = ExperimentService(authService: authService)
    }
}

@main
struct ToraApp: App {
    @StateObject private var services = AppServices()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environmentObject(services.authService)
                .environmentObject(services.workspaceService)
                .environmentObject(services.experimentService)
                .applyAppTheme()
                .task {
                    await refreshAuthTokenIfNeeded()
                }
                .onChange(of: scenePhase) {
                    guard scenePhase == .active else { return }
                    Task { await refreshAuthTokenIfNeeded() }
                }
        }
    }

    private func refreshAuthTokenIfNeeded() async {
        _ = try? await services.authService.getUserToken(skew: 180)
    }
}
