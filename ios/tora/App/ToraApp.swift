import SwiftUI

@main
struct ToraApp: App {
    @StateObject private var authService: AuthService
    @StateObject private var workspaceService: WorkspaceService
    @StateObject private var experimentService: ExperimentService
    @Environment(\.scenePhase) private var scenePhase

    init() {
        let authService = AuthService()
        let workspaceService = WorkspaceService(authService: authService)
        let experimentService = ExperimentService(authService: authService)

        _authService = StateObject(wrappedValue: authService)
        _workspaceService = StateObject(wrappedValue: workspaceService)
        _experimentService = StateObject(wrappedValue: experimentService)
    }

    var body: some Scene {
        WindowGroup {
            AppRootView()
                .environmentObject(authService)
                .environmentObject(workspaceService)
                .environmentObject(experimentService)
                .applyAppTheme()
                .task {
                    _ = try? await authService.getUserToken(skew: 180)
                }
                .onChange(of: scenePhase == .active) { _, isActive in
                    guard isActive else { return }
                    Task {
                        _ = try? await authService.getUserToken(skew: 180)
                    }
                }
        }
    }
}
