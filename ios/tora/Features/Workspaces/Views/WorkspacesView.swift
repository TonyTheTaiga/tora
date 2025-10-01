import SwiftUI

@MainActor
struct WorkspacesView: View {
    enum Route: Hashable {
        case experiment(String)
    }

    @EnvironmentObject private var workspaceService: WorkspaceService
    @StateObject private var viewModel: WorkspacesViewModel

    @MainActor
    init(viewModel: WorkspacesViewModel? = nil) {
        _viewModel = StateObject(wrappedValue: viewModel ?? WorkspacesViewModel())
    }

    var body: some View {
        content
            .navigationTitle("Workspaces")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: Route.self) { route in
                switch route {
                case .experiment(let id):
                    if let index = viewModel.experiments.firstIndex(where: { $0.id == id }) {
                        let binding = Binding(
                            get: { viewModel.experiments[index] },
                            set: { viewModel.updateExperiment($0) }
                        )
                        ExperimentDetailView(experiment: binding)
                    } else {
                        Text("Experiment not found")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .sheet(isPresented: $viewModel.isWorkspacePickerPresented) {
                WorkspacePickerModal(
                    workspaces: viewModel.workspaces,
                    selectedWorkspace: viewModel.selectedWorkspace,
                    onWorkspaceSelected: { workspace in viewModel.select(workspace: workspace) },
                    onClose: { viewModel.isWorkspacePickerPresented = false }
                )
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
            }
            .onAppear {
                viewModel.configure(service: workspaceService)
                viewModel.onAppear()
            }
            .onReceive(workspaceService.$workspaces) { _ in
                viewModel.configure(service: workspaceService)
            }
    }

    @ViewBuilder
    private var content: some View {
        switch viewModel.loadState {
        case .idle, .loading:
            ProgressView("Loading workspaces…")
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        case .error(let message):
            VStack(spacing: 12) {
                Text("We hit a snag")
                    .font(.headline)
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Button("Try Again") {
                    Task { await viewModel.loadWorkspaces() }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .padding()
        case .empty:
            VStack(spacing: 12) {
                Text("No Workspaces yet")
                    .font(.headline)
                Text("Create a workspace in the web app to see it here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
        case .loaded:
            loadedContent
        }
    }

    private var loadedContent: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                WorkspaceSelector(
                    selectedWorkspace: viewModel.selectedWorkspace,
                    onOpenPicker: { viewModel.isWorkspacePickerPresented = true }
                )
                Spacer()
            }
            .padding(.horizontal)

            if viewModel.isLoadingExperiments {
                ProgressView("Loading experiments…")
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding()
            } else if viewModel.selectedWorkspace != nil, viewModel.experiments.isEmpty {
                Text("No experiments in this workspace.")
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
            } else if viewModel.selectedWorkspace != nil {
                List {
                    ForEach(viewModel.experiments) { experiment in
                        NavigationLink(value: Route.experiment(experiment.id)) {
                            WorkspaceExperimentRow(experiment: experiment)
                        }
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .refreshable {
                    await viewModel.refreshSelectedWorkspace()
                }
            }
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }
}

private struct WorkspaceExperimentRow: View {
    let experiment: Experiment

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(experiment.name)
                .font(.headline)
                .foregroundStyle(Color.custom.ctpText)
            if let description = experiment.description, !description.isEmpty {
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    let authService = AuthService()
    let workspaceService = WorkspaceService(authService: authService)
    let experimentService = ExperimentService(authService: authService)

    return NavigationStack {
        WorkspacesView(viewModel: WorkspacesViewModel())
    }
    .environmentObject(authService)
    .environmentObject(workspaceService)
    .environmentObject(experimentService)
}
