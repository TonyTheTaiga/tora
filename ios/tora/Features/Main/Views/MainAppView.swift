import SwiftUI

struct MainAppView: View {
    @EnvironmentObject var authService: AuthService

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Welcome to Tora!")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                if let user = authService.currentUser {
                    Text("Hello, \(user.email)")
                        .font(.title2)
                        .foregroundColor(.secondary)
                }

                // Your main app content goes here
                Text("Main app content will go here")
                    .foregroundColor(.secondary)

                Spacer()

                Button("Logout") {
                    authService.logout()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            .navigationTitle("Tora")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Logout") {
                        authService.logout()
                    }
                }
            }
        }
    }
}

#Preview {
    MainAppView()
        .environmentObject(AuthService.shared)
}
