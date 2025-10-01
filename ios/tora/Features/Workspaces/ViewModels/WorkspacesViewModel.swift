import Combine
import Foundation

@MainActor
final class WorkspacesViewModel: ObservableObject {
    enum LoadState: Equatable {
        case idle
        case loading
        case loaded
        case empty
        case error(String)
    }

    @Published private(set) var loadState: LoadState = .idle
    @Published private(set) var experiments: [Experiment] = []
    @Published private(set) var workspaces: [Workspace] = []
    @Published private(set) var isLoadingExperiments = false
    @Published var selectedWorkspace: Workspace?
    @Published var isWorkspacePickerPresented = false

    private var service: WorkspaceService?
    private var experimentCache: [String: [Experiment]] = [:]

    func configure(service: WorkspaceService) {
        if self.service !== service {
            self.service = service
        }
        applyWorkspaceList(service.workspaces)
    }

    func onAppear() {
        guard let service else { return }
        if case .idle = loadState {
            Task { await loadWorkspaces(using: service) }
        }
    }

    func loadWorkspaces() async {
        guard let service else { return }
        await loadWorkspaces(using: service)
    }

    func refreshWorkspaces() async {
        guard let service else { return }
        do {
            try await service.list()
            applyWorkspaceList(service.workspaces)
            loadState = workspaces.isEmpty ? .empty : .loaded
        } catch {
            loadState = .error(error.localizedDescription)
        }
    }

    func select(workspace: Workspace) {
        selectedWorkspace = workspace
        isWorkspacePickerPresented = false
        if let cached = experimentCache[workspace.id] {
            experiments = cached
        } else {
            Task { await loadExperiments(for: workspace) }
        }
    }

    func refreshSelectedWorkspace() async {
        guard let workspace = selectedWorkspace else {
            await loadWorkspaces()
            return
        }
        await loadExperiments(for: workspace, forceReload: true)
    }

    func updateExperiment(_ experiment: Experiment) {
        guard let index = experiments.firstIndex(where: { $0.id == experiment.id }) else { return }
        experiments[index] = experiment
        if let selected = selectedWorkspace {
            experimentCache[selected.id] = experiments
        }
    }

    private func loadWorkspaces(using service: WorkspaceService) async {
        loadState = .loading
        do {
            try await service.list()
            applyWorkspaceList(service.workspaces)
            loadState = workspaces.isEmpty ? .empty : .loaded
            if let selected = selectedWorkspace {
                await loadExperiments(for: selected)
            }
        } catch {
            loadState = .error(error.localizedDescription)
        }
    }

    private func loadExperiments(for workspace: Workspace, forceReload: Bool = false) async {
        guard let service else { return }
        isLoadingExperiments = true
        do {
            if !forceReload, let existing = experimentCache[workspace.id] {
                experiments = existing
                isLoadingExperiments = false
                return
            }
            let fetched = try await service.listExperiments(for: workspace.id)
            experimentCache[workspace.id] = fetched
            experiments = fetched
            isLoadingExperiments = false
        } catch {
            isLoadingExperiments = false
            loadState = .error(error.localizedDescription)
        }
    }

    private func applyWorkspaceList(_ newValue: [Workspace]) {
        workspaces = newValue
        if let selected = selectedWorkspace,
            newValue.contains(where: { $0.id == selected.id }) == false
        {
            selectedWorkspace = nil
            experiments = []
        }
        updatePickerVisibility()
    }

    private func updatePickerVisibility() {
        isWorkspacePickerPresented = selectedWorkspace == nil && !workspaces.isEmpty
    }
}
