import SwiftData
import SwiftUI

@main
struct iosApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .modalBackground()
                .preferredColorScheme(.light)
                .modelContainer(for: UserSession.self)
        }
    }
}
