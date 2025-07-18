import SwiftData
import SwiftUI

struct WorkspacesView: View {
    @EnvironmentObject private var workspaceService: WorkspaceService
    @EnvironmentObject private var authService: AuthService
    @State private var workspaces: [Workspace] = []
    @State private var isLoading = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            VStack {
                if isLoading {
                    ProgressView("Fetching workspaces...")
                } else if let errorMessage = errorMessage {
                    VStack {
                        Text("Error")
                            .font(.headline)
                            .foregroundColor(.red)
                        Text(errorMessage)
                            .foregroundColor(.red)
                    }
                } else {
                    List(workspaces) { workspace in
                        VStack(alignment: .leading) {
                            Text(workspace.name)
                                .font(.headline)
                            Text(workspace.description)
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                    }
                }
            }
            .navigationTitle("Workspaces")
            .onAppear {
                fetchWorkspaces()
            }
        }
    }

    private func fetchWorkspaces() {
        isLoading = true
        errorMessage = nil
        Task {
            do {
                self.workspaces = try await workspaceService.list()
            } catch {
                self.errorMessage = error.localizedDescription
            }
            self.isLoading = false
        }
    }
}

#Preview {
    WorkspacesView()
        .modelContainer(for: UserSession.self, inMemory: true)
        .environmentObject(AuthService.shared)
        .environmentObject(WorkspaceService(authService: AuthService.shared))
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
        .preferredColorScheme(.light)
}
