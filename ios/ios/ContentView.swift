import SwiftUI

struct ContentView: View {
  @State private var logoScale: CGFloat = 0.8
  @State private var logoOpacity: Double = 0.0
  @State private var subtitleOffset: CGFloat = 20
  @State private var subtitleOpacity: Double = 0.0
  @State private var buttonOffset: CGFloat = 30
  @State private var buttonOpacity: Double = 0.0
  @State private var buttonPressed = false

  var body: some View {
    GeometryReader { geometry in
      VStack(spacing: 0) {
        Spacer()

        ToraLogo()
          .frame(width: geometry.size.width * 0.6)
          .scaleEffect(logoScale)
          .opacity(logoOpacity)
          .animation(.spring(response: 0.8, dampingFraction: 0.6), value: logoScale)
          .animation(.easeOut(duration: 0.6), value: logoOpacity)

        Spacer()
          .frame(height: geometry.size.height * 0.05)

        // Scrolling subtitle
        ScrollingSubtitle()
          .frame(height: 30)
          .opacity(subtitleOpacity)
          .animation(.easeOut(duration: 0.8).delay(0.3), value: subtitleOpacity)

        Spacer()
          .frame(height: geometry.size.height * 0.05)

        Button {
        } label: {
          Text("login")
            .font(.dynamicInter(16, weight: .medium, relativeTo: .body))
            .foregroundColor(buttonPressed ? Color.ctpBase : Color.ctpBlue)
            .padding(.horizontal, 32)
            .padding(.vertical, 12)
            .background(
              RoundedRectangle(cornerRadius: 0)
                .fill(buttonPressed ? Color.ctpBlue : Color.clear)
                .animation(.easeInOut(duration: 0.15), value: buttonPressed)
            )
            .overlay(
              RoundedRectangle(cornerRadius: 0)
                .stroke(Color.ctpBlue, lineWidth: 1)
            )
        }
        .scaleEffect(buttonPressed ? 0.95 : 1.0)
        .offset(y: buttonOffset)
        .opacity(buttonOpacity)
        .animation(.easeOut(duration: 0.8).delay(0.6), value: buttonOffset)
        .animation(.easeOut(duration: 0.8).delay(0.6), value: buttonOpacity)
        .animation(.easeInOut(duration: 0.1), value: buttonPressed)
        .onTapGesture {
          withAnimation(.easeInOut(duration: 0.1)) {
            buttonPressed = true
          }
          DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            withAnimation(.easeInOut(duration: 0.1)) {
              buttonPressed = false
            }
          }
        }

        Spacer()
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .padding(.horizontal, 20)
      .onAppear {
        // Trigger animations on appear
        withAnimation {
          logoScale = 1.0
          logoOpacity = 1.0
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
          withAnimation {
            subtitleOffset = 0
            subtitleOpacity = 1.0
          }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
          withAnimation {
            buttonOffset = 0
            buttonOpacity = 1.0
          }
        }
      }
    }
    .background(Color.ctpBase)
  }
}

struct ScrollingSubtitle: View {
  @State private var offset: CGFloat = 0
  private let text = "A Modern Experiment Tracker • "
  private let spacing: CGFloat = 0

  var body: some View {
    GeometryReader { geometry in
      HStack(spacing: spacing) {
        ForEach(0..<10, id: \.self) { _ in
          Text(text)
            .font(.dynamicInter(18, weight: .medium, relativeTo: .title2))
            .foregroundColor(Color.ctpSubtext1)
            .fixedSize()
        }
      }
      .offset(x: offset)
      .clipped()
      .onAppear {
        let textWidth = estimateTextWidth()

        withAnimation(.linear(duration: textWidth / 40).repeatForever(autoreverses: false)) {
          offset = -textWidth
        }
      }
    }
  }

  private func estimateTextWidth() -> CGFloat {
    return CGFloat(text.count) * 11  // Approximate character width for one instance
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
