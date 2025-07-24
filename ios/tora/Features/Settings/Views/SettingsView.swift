import SwiftUI

struct SettingsView: View {
    // MARK: - Properties

    @EnvironmentObject private var authService: AuthService

    // MARK: - Body

    var body: some View {
        Button("Logout") {
            Task {
                logout()
            }
        }
    }

    // MARK: - Private Methods

    private func logout() {
        authService.logout()
    }
}

// MARK: - Preview

#Preview {
    SettingsView()
}
