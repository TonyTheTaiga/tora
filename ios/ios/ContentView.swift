import SwiftUI

struct ContentView: View {
    @State private var isMaximized = false

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(spacing: 32) {
                    ToraLogo()
                        .frame(width: 200, height: 60)

                    VStack(spacing: 16) {
                        Text("A Modern Experiment Tracker")
                            .font(.system(size: 28, weight: .bold, design: .monospaced))
                            .foregroundColor(.primary)
                            .multilineTextAlignment(.center)

                        Rectangle()
                            .fill(Color.ctpBlue)
                            .frame(width: 60, height: 2)
                    }
                }
                .frame(maxWidth: .infinity, minHeight: geometry.size.height)
                .padding(.horizontal, 20)
            }
        }
        .background(Color.ctpBase)
    }
}

struct ToraLogo: View {
    var body: some View {
        ToraIcon()
            .fill(Color.ctpBlue)
            .aspectRatio(357.41/109.34, contentMode: ContentMode.fit)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
