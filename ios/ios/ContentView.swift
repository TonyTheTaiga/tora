import SwiftUI

struct ContentView: View {
  var body: some View {
    GeometryReader { geometry in
      ScrollView {
        VStack(spacing: geometry.size.height * 0.04) {
          ToraLogo()
            .frame(width: geometry.size.width * 0.75)

          VStack(spacing: geometry.size.height * 0.02) {
            Text("Pure Speed. Pure Insight.")
              .font(
                .system(
                  size: min(geometry.size.width * 0.08, geometry.size.height * 0.045),
                  weight: .bold, design: .monospaced)
              )
              .foregroundColor(Color.ctpText)
              .multilineTextAlignment(.center)
              .lineLimit(1)
              .minimumScaleFactor(0.6)

            Text("A Modern Experiment Tracker")
              .font(
                .system(
                  size: min(geometry.size.width * 0.06, geometry.size.height * 0.035),
                  weight: .bold, design: .monospaced)
              )
              .foregroundColor(Color.ctpText)
              .multilineTextAlignment(.center)
              .lineLimit(2)
              .minimumScaleFactor(0.7)

            Rectangle()
              .fill(Color.ctpBlue)
              .frame(width: geometry.size.width * 0.15, height: 2)
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
  private let logoAspectRatio: CGFloat = 357.41 / 109.34

  var body: some View {
    ToraIcon()
      .fill(Color.ctpBlue)
      .aspectRatio(logoAspectRatio, contentMode: ContentMode.fit)
  }
}

struct ContentView_Previews: PreviewProvider {
  static var previews: some View {
    ContentView()
  }
}
