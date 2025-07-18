import SwiftUI

struct ExperimentsView: View {
    var body: some View {
        NavigationStack {
        }
        .navigationTitle("Experiments")
        .modalBackground()
    }
}

#Preview {
    ExperimentsView()
        .preferredColorScheme(.light)
}
