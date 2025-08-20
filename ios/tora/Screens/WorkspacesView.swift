import SwiftUI

struct WorkspacesView: View {
    // MARK: - Properties

    @EnvironmentObject private var workspaceService: WorkspaceService
    @EnvironmentObject private var experimentService: ExperimentService
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var workspaceExperiments: [String: [Experiment]] = [:]
    @State private var selectedWorkspace: Workspace?
    @State private var isLoadingExperiments = false
    @State private var experiments: [Experiment] = []
    @State private var showingWorkspaceModal = false

    // MARK: - Body

    var body: some View {
        ZStack(alignment: .center) {
            // Main content
            Group {
                if isLoading {
                    ProgressView("Loading workspaces...")
                } else if let errorMessage = errorMessage {
                    VStack {
                        Text("Error")
                            .font(.headline)
                        Text(errorMessage)
                            .font(.caption)
                        Button("Try Again") { fetchWorkspaces() }
                    }
                } else if workspaceService.workspaces.isEmpty {
                    Text("No Workspaces")
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            WorkspaceSelector(
                                selectedWorkspace: selectedWorkspace,
                                onOpenPicker: { showingWorkspaceModal = true }
                            )
                            Spacer()
                            Button(action: {
                                if let ws = selectedWorkspace {
                                    fetchExperiments(for: ws)
                                } else {
                                    fetchWorkspaces()
                                }
                            }) {
                                Image(systemName: "arrow.clockwise")
                                    .font(.body)
                            }
                            .disabled(isLoading || isLoadingExperiments)
                        }
                        .padding(.horizontal)

                        if isLoadingExperiments {
                            ProgressView("Loading experiments...")
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding()
                        } else if selectedWorkspace != nil, experiments.isEmpty {
                            Text("No experiments in this workspace.")
                                .foregroundStyle(.secondary)
                                .padding(.horizontal)
                        } else if selectedWorkspace != nil {
                            List(experiments) { experiment in
                                NavigationLink(destination: ExperimentsView(experimentId: experiment.id)) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(experiment.name)
                                            .font(.headline)
                                            .foregroundStyle(Color.custom.ctpText)
                                        if let desc = experiment.description, !desc.isEmpty {
                                            Text(desc)
                                                .font(.caption)
                                                .foregroundStyle(.secondary)
                                                .lineLimit(2)
                                        }
                                    }
                                    .padding(.vertical, 4)
                                }
                            }
                            .listStyle(.plain)
                            .scrollContentBackground(.hidden)
                        }
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
            }

            // True modal overlay
            if showingWorkspaceModal {
                Color.custom.ctpCrust.opacity(0.45)
                    .ignoresSafeArea()
                    .onTapGesture { showingWorkspaceModal = false }

                WorkspacePickerModal(
                    workspaces: workspaceService.workspaces,
                    selectedWorkspace: selectedWorkspace,
                    onWorkspaceSelected: { ws in selectWorkspace(ws) },
                    onClose: { showingWorkspaceModal = false }
                )
                .padding(24)
            }
        }
        .navigationTitle("Workspaces")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {}
        .onAppear(perform: {
            fetchWorkspaces()
        })
    }

    // MARK: - Private Methods

    private func fetchWorkspaces() {
        isLoading = true
        errorMessage = nil
        Task {
            defer { isLoading = false }

            do {
                try await workspaceService.list()
            } catch {
                self.errorMessage = error.localizedDescription
            }
        }
    }

    private func selectWorkspace(_ workspace: Workspace) {
        selectedWorkspace = workspace
        if let cached = workspaceExperiments[workspace.id] {
            experiments = cached
        } else {
            fetchExperiments(for: workspace)
        }
    }

    private func fetchExperiments(for workspace: Workspace) {
        isLoadingExperiments = true
        errorMessage = nil
        Task {
            do {
                let exps = try await workspaceService.listExperiments(for: workspace.id)
                await MainActor.run {
                    self.workspaceExperiments[workspace.id] = exps
                    self.experiments = exps
                    self.isLoadingExperiments = false
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isLoadingExperiments = false
                }
            }
        }
    }
}
