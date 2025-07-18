import SwiftUI

struct MainAppView: View {
    @EnvironmentObject var authService: AuthService

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                if let user = authService.currentUser {
                    Text("Hello, \(user.email)")
                        .font(.title2)
                        .foregroundColor(.secondary)
                }

                Spacer()
            }
            .padding()
            .navigationTitle("Tora")
        }
    }
}

#Preview {
    MainAppView()
        .environmentObject(AuthService.shared)
}
