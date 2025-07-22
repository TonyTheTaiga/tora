import SwiftData
import SwiftUI

struct WorkspacesView: View {
    // MARK: - Properties

    let onExperimentSelected: ((String) -> Void)
    @EnvironmentObject private var workspaceService: WorkspaceService
    @EnvironmentObject private var experimentService: ExperimentService
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var experiments: [String: [Experiment]] = [:]

    // MARK: - Body

    var body: some View {
        Group {
            if isLoading {
                ProgressView("Loading workspaces...")
            } else if let errorMessage = errorMessage {
                VStack {
                    Text("Error Loading Workspaces")
                        .font(.headline)
                    Text(errorMessage)
                        .font(.caption)
                    Button("Try Again") {
                        fetchWorkspaces()
                    }
                }
            } else if workspaceService.workspaces.isEmpty {
                Text("No Workspaces")
            } else {
                ScrollView {
                    VStack {
                        ForEach(workspaceService.workspaces, id: \.id) {
                            workspace in
                            WorkspaceCard(
                                workspace: workspace,
                                experiments: experiments[workspace.id] ?? [],
                                onExperimentSelected: onExperimentSelected
                            )
                        }
                    }
                }
            }
        }
        .navigationTitle("Workspaces")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: { fetchWorkspaces() }) {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(isLoading)
            }
        }
        .onAppear(perform: {
            fetchWorkspaces()
        })
    }

    // MARK: - Private Methods

    private func fetchWorkspaces() {
        isLoading = true
        errorMessage = nil
        Task {
            do {
                try await workspaceService.list()
                for workspace in workspaceService.workspaces {
                    experiments[workspace.id] =
                        try await workspaceService.listExperiments(
                            for: workspace.id
                        )
                }
                self.isLoading = false
            } catch {
                self.errorMessage = error.localizedDescription
            }
        }
    }
}
