import SwiftData
import SwiftUI

struct WorkspaceRow: View {
    let workspace: Workspace
    @State private var showingExperiments = false

    var body: some View {
        Button(action: {
            showingExperiments = true
        }) {
            VStack(alignment: .leading, spacing: 6) {
                Text(workspace.name)
                    .font(.headline)
                    .foregroundColor(.primary)

                Text(workspace.description ?? "No description")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Text("Role: \(workspace.role)")
                    .font(.caption)
                    .foregroundColor(.accentColor)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 6)
            .padding(.horizontal, 16)
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
            VStack {
                if isLoading {
                    VStack(spacing: 16) {
                        ProgressView()
                        Text("Loading experiments...")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let errorMessage = errorMessage {
                    VStack(spacing: 16) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundColor(.orange)
                        Text("Error loading experiments")
                            .font(.headline)
                        Text(errorMessage)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                        Button("Try Again") {
                            fetchExperiments()
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if experiments.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "flask")
                            .font(.largeTitle)
                            .foregroundColor(.secondary)
                        Text("No experiments found")
                            .font(.headline)
                        Text("This workspace doesn't have any experiments yet.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(experiments) { experiment in
                        ExperimentRow(experiment: experiment)
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle(workspace.name)
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
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
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(experiment.name)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Spacer()

                Text(formatDate(experiment.updatedAt))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            if let description = experiment.description, !description.isEmpty {
                Text(description)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
            }

            if !experiment.tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(experiment.tags, id: \.self) { tag in
                            Text(tag)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.accentColor.opacity(0.1))
                                .foregroundColor(.accentColor)
                                .cornerRadius(4)
                        }
                    }
                    .padding(.horizontal, 1)
                }
            }

            if !experiment.hyperparams.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 4) {
                        ForEach(experiment.hyperparams, id: \.key) { hyperparam in
                            Text("\(hyperparam.key): \(hyperparam.value.displayValue)")
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.gray.opacity(0.1))
                                .foregroundColor(.secondary)
                                .cornerRadius(4)
                        }
                    }
                    .padding(.horizontal, 1)
                }
            }

            if !experiment.availableMetrics.isEmpty {
                Text("Metrics: \(experiment.availableMetrics.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 8)
        .background(Color.gray.opacity(0.1))
        .cornerRadius(8)
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
                        WorkspaceRow(workspace: workspace)
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
}
