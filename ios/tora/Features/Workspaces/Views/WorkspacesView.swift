import SwiftData
import SwiftUI

struct WorkspaceRow: View {
    let workspace: Workspace
    @State private var showingExperiments = false

    var body: some View {
        Button(action: {
            showingExperiments = true
        }) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(workspace.name)
                        .font(.headline)
                        .foregroundColor(.primary)

                    Spacer()

                    Text(workspace.role)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.secondary.opacity(0.1))
                        .clipShape(Capsule())
                }

                if let description = workspace.description, !description.isEmpty {
                    Text(description)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .lineLimit(2)
                        .multilineTextAlignment(.leading)
                }

                HStack {
                    Image(systemName: "folder")
                        .foregroundColor(.secondary)
                        .font(.caption)

                    Text("Tap to view experiments")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Image(systemName: "chevron.right")
                        .foregroundColor(.secondary)
                        .font(.caption2)
                }
            }
            .padding()
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            .contentShape(Rectangle())
        }
        .buttonStyle(PlainButtonStyle())
        .sheet(isPresented: $showingExperiments) {
            WorkspaceExperimentsView(workspace: workspace)
        }
    }
}

struct WorkspaceExperimentsView: View {
    let workspace: Workspace
    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var experiments: [Experiment] = []
    @State private var isLoading = true
    @State private var errorMessage: String?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            Group {
                if isLoading {
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading experiments...")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage = errorMessage {
                    ContentUnavailableView {
                        Label("Error Loading Experiments", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(errorMessage)
                    } actions: {
                        ToraButton("Try Again", style: .primary) {
                            fetchExperiments()
                        }
                    }
                } else if experiments.isEmpty {
                    ContentUnavailableView {
                        Label("No Experiments", systemImage: "flask")
                    } description: {
                        Text("This workspace doesn't have any experiments yet.")
                    }
                } else {
                    List(experiments) { experiment in
                        ExperimentRow(experiment: experiment)
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle(workspace.name)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    ToraToolbarButton(systemImage: "xmark") {
                        dismiss()
                    }
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
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}

struct ExperimentRow: View {
    let experiment: Experiment

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(experiment.name)
                    .font(.headline)
                    .foregroundColor(.primary)

                Spacer()

                Text(formatDate(experiment.updatedAt))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if let description = experiment.description, !description.isEmpty {
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            if !experiment.tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(experiment.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.blue.opacity(0.1))
                                .foregroundColor(.blue)
                                .clipShape(Capsule())
                        }
                    }
                    .padding(.horizontal, 1)
                }
            }

            if !experiment.hyperparams.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(experiment.hyperparams, id: \.key) { hyperparam in
                            Text("\(hyperparam.key): \(hyperparam.value.displayValue)")
                                .font(.caption)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(.secondary.opacity(0.1))
                                .foregroundColor(.secondary)
                                .clipShape(Capsule())
                        }
                    }
                    .padding(.horizontal, 1)
                }
            }

            if !experiment.availableMetrics.isEmpty {
                Text("Metrics: \(experiment.availableMetrics.joined(separator: ", "))")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter.localizedString(for: date, relativeTo: Date())
    }
}

struct WorkspacesView: View {
    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var workspaces: [Workspace] = []
    @State private var isLoading = true
    @State private var errorMessage: String?

    var body: some View {
        NavigationView {
            Group {
                if isLoading {
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading workspaces...")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage = errorMessage {
                    ContentUnavailableView {
                        Label("Error Loading Workspaces", systemImage: "exclamationmark.triangle")
                    } description: {
                        Text(errorMessage)
                    } actions: {
                        ToraButton("Try Again", style: .primary) {
                            fetchWorkspaces()
                        }
                    }
                } else if workspaces.isEmpty {
                    ContentUnavailableView {
                        Label("No Workspaces", systemImage: "folder")
                    } description: {
                        Text("No workspaces are available for your account.")
                    }
                } else {
                    List(workspaces) { workspace in
                        WorkspaceRow(workspace: workspace)
                    }
                    .listStyle(.insetGrouped)
                }
            }
            .navigationTitle("Workspaces")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    ToraToolbarButton(systemImage: "arrow.clockwise") {
                        fetchWorkspaces()
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
}
