import SwiftUI

struct SettingsView: View {
    @Environment(\.modelContext) private var context
    @EnvironmentObject private var authService: AuthService

    var body: some View {
        Button("Logout") {
            Task {
                logOut()
            }
        }
    }

    func logOut() {
        authService.logout()
    }
}

#Preview {
    SettingsView()
}
