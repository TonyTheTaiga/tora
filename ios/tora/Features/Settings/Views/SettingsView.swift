import SwiftUI

struct SettingsView: View {
    // MARK: - Properties

    @Environment(\.modelContext) private var context
    @EnvironmentObject private var authService: AuthService

    // MARK: - Body

    var body: some View {
        Button("Logout") {
            Task {
                logOut()
            }
        }
    }

    // MARK: - Public Methods

    func logOut() {
        authService.logout()
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
