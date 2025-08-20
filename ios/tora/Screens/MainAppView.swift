import SwiftUI

struct MainAppView: View {
    // MARK: - Properties

    @State private var selectedTab: Tabs = .workspaces

    // MARK: - Tabs

    enum Tabs: Equatable, Hashable {
        case workspaces
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
                    WorkspacesView()
                }
            }

            Tab("Settings", systemImage: "gearshape.2", value: .settings) {
                SettingsView()
            }
        }
        .tint(Color.custom.ctpBlue)
        .tabBarMinimizeBehavior(.onScrollDown)
    }
}
