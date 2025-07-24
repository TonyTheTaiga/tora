import SwiftData
import SwiftUI

struct MainAppView: View {
    // MARK: - Properties

    @State private var selectedTab: Tabs = .workspaces
    @State private var selectedExperimentId: String?

    // MARK: - Tabs

    enum Tabs: Equatable, Hashable {
        case workspaces
        case experiments
        case settings
    }

    // MARK: - Body

    var body: some View {
        TabView(selection: $selectedTab) {
            Tab(
                "Workspaces",
                systemImage: "folder",
                value: .workspaces
            ) {
                NavigationStack {
                    WorkspacesView(onExperimentSelected: { experimentId in
                        selectedExperimentId = experimentId
                        selectedTab = .experiments
                    })
                }
            }

            Tab("Experiments", systemImage: "receipt", value: .experiments) {
                NavigationStack {
                    ExperimentsView(experimentId: selectedExperimentId)
                }
            }

            Tab("Settings", systemImage: "gearshape.2", value: .settings) {
                SettingsView()
            }
        }
        .accentColor(.blue)
    }
}
