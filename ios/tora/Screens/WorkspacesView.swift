import SwiftUI

struct WorkspacesView: View {
    // MARK: - Properties

    @EnvironmentObject private var workspaceService: WorkspaceService
    @State private var isLoading = true
    @State private var errorMessage: String?
    @State private var workspaceExperiments: [String: [Experiment]] = [:]
    @State private var selectedWorkspace: Workspace?
    @State private var isLoadingExperiments = false
    @State private var experiments: [Experiment] = []
    @State private var showingWorkspaceModal = false

    enum Route: Hashable {
        case experiment(String)
    }

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
                            List {
                                ForEach(experiments) { experiment in
                                    NavigationLink(value: Route.experiment(experiment.id)) {
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
                            }
                            .listStyle(.plain)
                            .scrollContentBackground(.hidden)
                            .refreshable {
                                await refreshData()
                            }
                        }
                        Spacer(minLength: 0)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
            }

        }
        .navigationTitle("Workspaces")
        .navigationBarTitleDisplayMode(.inline)
        .navigationDestination(for: Route.self) { route in
            switch route {
            case .experiment(let id):
                if let idx = experiments.firstIndex(where: { $0.id == id }) {
                    ExperimentsView(experiment: $experiments[idx])
                } else {
                    Text("Experiment not found")
                        .foregroundStyle(.secondary)
                }
            }
        }
        .toolbar {}
        .sheet(isPresented: $showingWorkspaceModal) {
            WorkspacePickerModal(
                workspaces: workspaceService.workspaces,
                selectedWorkspace: selectedWorkspace,
                onWorkspaceSelected: { ws in selectWorkspace(ws) },
                onClose: { showingWorkspaceModal = false }
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
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
                await MainActor.run {
                    isLoading = false
                    maybePresentWorkspacePicker()
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
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

    private func refreshData() async {
        if let ws = selectedWorkspace {
            do {
                let exps = try await workspaceService.listExperiments(for: ws.id)
                await MainActor.run {
                    self.workspaceExperiments[ws.id] = exps
                    self.experiments = exps
                }
            } catch {
                // keep existing experiments; optionally set errorMessage
            }
        } else {
            do {
                try await workspaceService.list()
                await MainActor.run {
                    maybePresentWorkspacePicker()
                }
            } catch {
                // ignore in refresh context
            }
        }
    }

    @MainActor
    private func maybePresentWorkspacePicker() {
        if selectedWorkspace == nil && !workspaceService.workspaces.isEmpty {
            showingWorkspaceModal = true
        }
    }
}
