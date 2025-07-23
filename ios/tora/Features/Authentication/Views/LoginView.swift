import SwiftData
import SwiftUI

// MARK: - Login Form Sheet

struct LoginFormSheet: View {
    // MARK: - Properties

    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @EnvironmentObject var authService: AuthService
    @State private var email = ""
    @State private var password = ""
    @State private var errorMessage = ""
    @State private var isLoading = false
    @State private var showError = false
    @FocusState private var isFocused: Field?

    enum Field {
        case email, password
    }

    // MARK: - Body

    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("Sign In")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                TextField("Email", text: $email)
                    .focused($isFocused, equals: .email)
                    .onSubmit {
                        isFocused = .password
                    }
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                SecureField("Password", text: $password)
                    .focused($isFocused, equals: .password)
                    .onSubmit {
                        if !isLoading {
                            Task { await signIn() }
                        }
                    }
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                if showError {
                    Text(errorMessage)
                        .foregroundColor(.red)
                        .font(.caption)
                        .multilineTextAlignment(.center)
                }

                Button(isLoading ? "Signing In..." : "Sign In") {
                    Task { await signIn() }
                }
                .disabled(isLoading || email.isEmpty || password.isEmpty)

                Spacer()
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark")
                    }
                }
            }
        }
    }

    // MARK: - Private Methods

    private func signIn() async {
        isLoading = true
        showError = false
        errorMessage = ""

        defer {
            isLoading = false
        }

        do {
            let _ = try await authService.login(
                email: email,
                password: password
            )
            dismiss()
        } catch {
            if let authError = error as? LocalizedError {
                errorMessage =
                    authError.errorDescription ?? "Authentication failed"
            } else {
                errorMessage = "An unexpected error occurred. Please try again."
            }
            showError = true
        }
    }
}

// MARK: - Login View

struct LoginView: View {
    // MARK: - Properties

    @State private var logoScale: CGFloat = 0.8
    @State private var logoOpacity: Double = 0.0
    @State private var subtitleOffset: CGFloat = 20
    @State private var subtitleOpacity: Double = 0.0
    @State private var buttonOffset: CGFloat = 30
    @State private var buttonOpacity: Double = 0.0
    @State private var loginSheetShown: Bool = false

    // MARK: - Body

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Spacer()

                Image("ToraLogo")
                    .renderingMode(.template)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: geometry.size.width * 0.6)
                    .foregroundColor(Color.custom.ctpBlue)
                    .scaleEffect(logoScale)
                    .opacity(logoOpacity)
                    .animation(
                        .spring(response: 0.8, dampingFraction: 0.6),
                        value: logoScale
                    )
                    .animation(.easeOut(duration: 0.6), value: logoOpacity)

                Spacer()
                    .frame(height: geometry.size.height * 0.05)

                ScrollingSubtitle()
                    .frame(height: 30)
                    .opacity(subtitleOpacity)
                    .animation(
                        .easeOut(duration: 0.8).delay(0.3),
                        value: subtitleOpacity
                    )

                Spacer()
                    .frame(height: geometry.size.height * 0.05)

                Button("Login") {
                    loginSheetShown = true
                }
                .offset(y: buttonOffset)
                .opacity(buttonOpacity)
                .animation(
                    .easeOut(duration: 0.8).delay(0.6),
                    value: buttonOffset
                )
                .animation(
                    .easeOut(duration: 0.8).delay(0.6),
                    value: buttonOpacity
                )
                .sheet(isPresented: $loginSheetShown) {
                    LoginFormSheet()
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
    }
}

// MARK: - Scrolling Subtitle

struct ScrollingSubtitle: View {
    // MARK: - Properties

    @State private var offset: CGFloat = 0
    @State private var textWidth: CGFloat = 0

    private let text = "A Modern Experiment Tracker • "
    private let spacing: CGFloat = 0
    private let scrollSpeed: CGFloat = 40  // pixels per second

    // MARK: - Body

    var body: some View {
        GeometryReader { geometry in
            let dynamicFontSize = min(max(geometry.size.width * 0.045, 16), 28)

            HStack(spacing: spacing) {
                ForEach(
                    0..<max(
                        2,
                        textWidth > 0
                            ? Int(ceil(geometry.size.width / textWidth)) + 1 : 3
                    ),
                    id: \.self
                ) { _ in
                    Text(text)
                        .font(
                            .system(
                                size: dynamicFontSize,
                                weight: .bold,
                                design: .default
                            )
                        )
                        .foregroundColor(.secondary)
                        .fixedSize()
                        .background(
                            GeometryReader { textGeometry in
                                Color.clear.onAppear {
                                    if textWidth == 0 {
                                        textWidth = textGeometry.size.width
                                    }
                                }
                            }
                        )
                }
            }
            .offset(x: offset)
            .clipped()
            .onAppear {
                startScrolling()
            }
            .onChange(of: dynamicFontSize) {
                textWidth = 0
                offset = 0
            }
        }
    }

    // MARK: - Private Methods

    private func startScrolling() {
        guard textWidth > 0 else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                startScrolling()
            }
            return
        }

        withAnimation(
            .linear(duration: textWidth / scrollSpeed).repeatForever(
                autoreverses: false
            )
        ) {
            offset = -textWidth
        }
    }
}

// MARK: - Tora Logo

struct ToraLogo: View {
    // MARK: - Properties

    private let logoAspectRatio: CGFloat = 357.41 / 109.34

    // MARK: - Body

    var body: some View {
        Image("ToraLogo")
            .renderingMode(.template)
            .resizable()
            .aspectRatio(logoAspectRatio, contentMode: .fit)
            .foregroundColor(Color.custom.ctpBlue)
    }
}

// MARK: - Preview

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        LoginView()
    }
}
