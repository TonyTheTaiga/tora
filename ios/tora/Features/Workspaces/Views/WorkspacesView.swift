import SwiftData
import SwiftUI

struct WorkspaceSimpleRow: View {
    let workspace: Workspace

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(workspace.name)
                .font(.headline)

            Text(workspace.description ?? "No description")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Text("Role: \(workspace.role)")
                .font(.caption)
                .foregroundColor(.accentColor)
        }
        .padding(.vertical, 6)
    }
}

struct WorkspacesView: View {
    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var workspaces: [Workspace] = []
    @State private var isLoading = true
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            VStack {
                if isLoading {
                    ProgressView("Loading Workspaces...")
                } else if let errorMessage = errorMessage {
                    Text("Error: \(errorMessage)")
                        .foregroundColor(.red)
                        .padding()
                } else if workspaces.isEmpty {
                    Text("No Workspaces Found")
                        .foregroundColor(.secondary)
                } else {
                    List(workspaces) { workspace in
                        NavigationLink(destination: Text("Details for \(workspace.name)")) {
                            WorkspaceSimpleRow(workspace: workspace)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .navigationTitle("Workspaces")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: fetchWorkspaces) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(isLoading)
                }
            }
            .onAppear(perform: fetchWorkspaces)
        }
        .navigationViewStyle(.stack)
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
            let authService = AuthService.shared
            authService.isAuthenticated = true
            authService.currentUser = UserSession(
                id: "preview-user",
                email: "preview@tora.com",
                auth_token: "token",
                refresh_token: "refresh",
                expiresIn: Date(),
                expiresAt: Date().addingTimeInterval(3600),
                tokenType: "Bearer"
            )
        }
}
