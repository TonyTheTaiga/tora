import SwiftUI

struct LoginView: View {
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var isSignInPresented = false

    var body: some View {
        content
            .sheet(isPresented: $isSignInPresented) {
                LoginFormSheet()
                    .presentationDetents([.medium, .large])
                    .presentationDragIndicator(.visible)
                    .presentationCompactAdaptation(.popover)
                    .onDisappear { isSignInPresented = false }
            }
    }

    @ViewBuilder
    private var content: some View {
        VStack(spacing: 0) {
            header

            Spacer()

            Image("ToraLogo")
                .renderingMode(.template)
                .resizable()
                .scaledToFit()
                .frame(width: UIDevice.current.userInterfaceIdiom == .pad ? 300 : 280)
                .offset(x: (horizontalSizeClass == .regular) ? 30 : 16)
                .foregroundColor(Color.custom.ctpBlue)

            Spacer().frame(height: 24)

            message
                .padding(.horizontal)

            Spacer().frame(height: 24)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.horizontal, 20)
    }

    private var header: some View {
        HStack(spacing: 12) {
            Spacer()
            ToraButton(
                "sign in",
                size: .small,
                backgroundColor: Color.custom.ctpSurface0.opacity(0.20),
                borderColor: Color.custom.ctpSurface0.opacity(0.30),
                textColor: Color.custom.ctpBlue,
                systemImage: "arrow.right"
            ) {
                isSignInPresented = true
            }
        }
        .padding(.horizontal, 20)
        .padding(.top, 12)
    }

    private var message: some View {
        VStack(spacing: 8) {
            ViewThatFits(in: .horizontal) {
                Text("Pure Speed. Pure Insight.")
                    .font(.system(.largeTitle, design: .monospaced))
                    .fontWeight(.bold)
                    .lineLimit(1)
                Text("Pure Speed. Pure Insight.")
                    .font(.system(.title, design: .monospaced))
                    .fontWeight(.bold)
                    .lineLimit(1)
                Text("Pure Speed. Pure Insight.")
                    .font(.system(.title2, design: .monospaced))
                    .fontWeight(.bold)
                    .lineLimit(1)
            }
            .multilineTextAlignment(.center)
            .foregroundColor(Color.custom.ctpText)
            .allowsTightening(true)

            Text("A New Experiment Tracker")
                .font(.system(.title3, design: .monospaced))
                .multilineTextAlignment(.center)
                .foregroundColor(Color.custom.ctpSubtext1)
        }
    }
}

// MARK: - Preview

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        let authService = AuthService()
        return LoginView()
            .environmentObject(authService)
    }
}
