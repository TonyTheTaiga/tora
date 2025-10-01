import SwiftUI

/// Root view that routes between authenticated and unauthenticated flows.
struct AppRootView: View {
    @EnvironmentObject private var authService: AuthService

    var body: some View {
        Group {
            switch authService.state {
            case .authenticated:
                MainAppView()
            case .unauthenticated(let message):
                if let message, !message.isEmpty {
                    VStack(spacing: 12) {
                        Text(message)
                            .font(.callout)
                            .foregroundStyle(.red)
                        LoginView()
                    }
                    .padding()
                } else {
                    LoginView()
                }
            }
        }
    }
}

#Preview {
    let authService = AuthService()
    let workspaceService = WorkspaceService(authService: authService)
    let experimentService = ExperimentService(authService: authService)

    return AppRootView()
        .environmentObject(authService)
        .environmentObject(workspaceService)
        .environmentObject(experimentService)
}
