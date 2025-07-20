import SwiftData
import SwiftUI

struct WorkspaceRow: View {
    let workspace: Workspace
    let onExperimentSelected: ((String) -> Void)?

    var body: some View {
        NavigationLink {
            WorkspaceExperimentsView(workspace: workspace, onExperimentSelected: onExperimentSelected)
        } label: {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(workspace.name)
                        .font(.headline)

                    Spacer()

                    Text(workspace.role)
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.1))
                        .clipShape(Capsule())
                }

                if let description = workspace.description, !description.isEmpty {
                    Text(description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                }
            }
        }
    }
}

struct WorkspaceExperimentsView: View {
    let workspace: Workspace
    let onExperimentSelected: ((String) -> Void)?
    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var experiments: [Experiment] = []
    @State private var isLoading = true
    @State private var errorMessage: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        Group {
            if isLoading {
                ProgressView("Loading experiments...")
            } else if let errorMessage = errorMessage {
                VStack {
                    Text("Error Loading Experiments")
                        .font(.headline)
                    Text(errorMessage)
                        .font(.caption)
                    Button("Try Again") {
                        fetchExperiments()
                    }
                }
            } else if experiments.isEmpty {
                Text("No Experiments")
            } else {
                List(experiments) { experiment in
                    ExperimentRow(experiment: experiment, onExperimentSelected: onExperimentSelected)
                }
            }
        }
        .navigationTitle(workspace.name)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: { dismiss() }) {
                    Image(systemName: "xmark")
                }
            }
        }
        .onAppear {
            fetchExperiments()
        }
    }

    private func fetchExperiments() {
        isLoading = true
        errorMessage = nil

        Task {
            do {
                let fetchedExperiments = try await workspaceService.listExperiments(for: workspace.id)
                await MainActor.run {
                    self.experiments = fetchedExperiments
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    if let workspaceError = error as? WorkspaceErrors {
                        print("Caught workspace error: \(workspaceError)")
                    }
                    print("Error fetching experiments: \(error)")
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}

struct ExperimentRow: View {
    let experiment: Experiment
    let onExperimentSelected: ((String) -> Void)?

    var body: some View {
        Button(
            action: {
                onExperimentSelected?(experiment.id)
            },
            label: {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(experiment.name)
                            .font(.headline)

                        Spacer()

                        Text(formatDate(experiment.updatedAt))
                            .font(.caption)
                    }

                    if let description = experiment.description, !description.isEmpty {
                        Text(description)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                    }
                }
            }
        )
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct WorkspacesView: View {
    let onExperimentSelected: ((String) -> Void)?
    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var workspaces: [Workspace] = []
    @State private var isLoading = true
    @State private var errorMessage: String?

    init(onExperimentSelected: ((String) -> Void)? = nil) {
        self.onExperimentSelected = onExperimentSelected
    }

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
            } else if workspaces.isEmpty {
                Text("No Workspaces")
            } else {
                List(workspaces) { workspace in
                    WorkspaceRow(workspace: workspace, onExperimentSelected: onExperimentSelected)
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
        .onAppear(perform: fetchWorkspaces)
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
        .environmentObject(WorkspaceService(authService: AuthService.shared))
}
