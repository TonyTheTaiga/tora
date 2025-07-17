import SwiftUI

struct ContentView: View {
    @State private var logoScale: CGFloat = 0.8
    @State private var logoOpacity: Double = 0.0
    @State private var subtitleOffset: CGFloat = 20
    @State private var subtitleOpacity: Double = 0.0
    @State private var buttonOffset: CGFloat = 30
    @State private var buttonOpacity: Double = 0.0
    @State private var buttonPressed = false
    @State private var loginSheetShown: Bool = false

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

                ScrollingSubtitle()
                    .frame(height: 30)
                    .opacity(subtitleOpacity)
                    .animation(.easeOut(duration: 0.8).delay(0.3), value: subtitleOpacity)

                Spacer()
                    .frame(height: geometry.size.height * 0.05)

                Button {
                } label: {
                    let buttonFontSize = min(max(geometry.size.width * 0.04, 14), 20)
                    let horizontalPadding = min(max(geometry.size.width * 0.08, 24), 48)
                    let verticalPadding = min(max(geometry.size.height * 0.015, 10), 16)

                    Text("login")
                        .font(
                            Font.dynamicInter(
                                buttonFontSize, weight: Font.Weight.medium, relativeTo: Font.TextStyle.body)
                        )
                        .foregroundColor(buttonPressed ? Color.ctpBase : Color.ctpBlue)
                        .padding(.horizontal, horizontalPadding)
                        .padding(.vertical, verticalPadding)
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
            let dynamicFontSize = min(max(geometry.size.width * 0.045, 16), 28)

            HStack(spacing: spacing) {
                ForEach(0..<10, id: \.self) { _ in
                    Text(text)
                        .font(
                            .dynamicInter(
                                dynamicFontSize, weight: Font.Weight.bold, relativeTo: Font.TextStyle.title2)
                        )
                        .foregroundColor(Color.ctpSubtext1)
                        .fixedSize()
                }
            }
            .offset(x: offset)
            .clipped()
            .onAppear {
                let textWidth = estimateTextWidth(fontSize: dynamicFontSize)

                withAnimation(.linear(duration: textWidth / 40).repeatForever(autoreverses: false)) {
                    offset = -textWidth
                }
            }
        }
    }

    private func estimateTextWidth(fontSize: CGFloat) -> CGFloat {
        let characterWidth = fontSize * 0.6  // Inter font has roughly 0.6 character width ratio
        return CGFloat(text.count) * characterWidth
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
