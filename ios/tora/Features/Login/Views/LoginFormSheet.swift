import SwiftUI

// MARK: - Login Form Sheet

struct LoginFormSheet: View {
    // MARK: - Properties

    @EnvironmentObject var authService: AuthService
    @Environment(\.dismiss) private var dismiss
    @State private var email = ""
    @State private var password = ""
    @State private var errorMessage = ""
    @State private var isLoading = false
    @FocusState private var isFocused: Field?

    enum Field {
        case email, password
    }

    // MARK: - Body

    var body: some View {
        ZStack {
            Color.clear
            VStack {
                Spacer(minLength: 16)
                VStack(spacing: 0) {
                    HStack(spacing: 8) {
                        Image(systemName: "person.circle.fill")
                            .font(.system(size: 20))
                            .foregroundColor(Color.custom.ctpMauve)
                        Text("sign in")
                            .font(.system(.title3, design: .monospaced))
                            .foregroundColor(Color.custom.ctpText)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        Rectangle()
                            .fill(Color.clear)
                            .overlay(
                                Rectangle()
                                    .frame(height: 1)
                                    .foregroundColor(Color.custom.ctpSurface0.opacity(0.6)),
                                alignment: .bottom
                            )
                    )

                    VStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("email address")
                                .font(.system(.footnote, design: .monospaced))
                                .foregroundColor(Color.custom.ctpSubtext0)

                            ZStack(alignment: .leading) {
                                HStack(spacing: 8) {
                                    Image(systemName: "envelope.fill")
                                        .foregroundColor(Color.custom.ctpSubtext0)
                                        .font(.system(size: 16))
                                    TextField("your.email@example.com", text: $email)
                                        .textContentType(.emailAddress)
                                        .keyboardType(.emailAddress)
                                        .textInputAutocapitalization(.never)
                                        .disableAutocorrection(true)
                                        .font(.system(.body, design: .monospaced))
                                        .onSubmit { isFocused = .password }
                                        .focused($isFocused, equals: .email)
                                }
                                .padding(.vertical, 10)
                                .padding(.horizontal, 12)
                            }
                            .background(Color.custom.ctpSurface0.opacity(0.2))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(
                                        isFocused == .email
                                            ? Color.custom.ctpBlue
                                            : Color.custom.ctpSurface0.opacity(0.6),
                                        lineWidth: 1
                                    )
                            )
                            .cornerRadius(8)
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            Text("password")
                                .font(.system(.footnote, design: .monospaced))
                                .foregroundColor(Color.custom.ctpSubtext0)

                            ZStack(alignment: .leading) {
                                HStack(spacing: 8) {
                                    Image(systemName: "lock.fill")
                                        .foregroundColor(Color.custom.ctpSubtext0)
                                        .font(.system(size: 16))
                                    SecureField("enter your password", text: $password)
                                        .textContentType(.password)
                                        .font(.system(.body, design: .monospaced))
                                        .onSubmit { if !isLoading { signIn() } }
                                        .focused($isFocused, equals: .password)
                                }
                                .padding(.vertical, 10)
                                .padding(.horizontal, 12)
                            }
                            .background(Color.custom.ctpSurface0.opacity(0.2))
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(
                                        isFocused == .password
                                            ? Color.custom.ctpMauve
                                            : Color.custom.ctpSurface0.opacity(0.6),
                                        lineWidth: 1
                                    )
                            )
                            .cornerRadius(8)
                        }

                        if errorMessage != "" {
                            Text(errorMessage)
                                .font(.system(.footnote, design: .monospaced))
                                .foregroundColor(Color.custom.ctpRed)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(10)
                                .background(Color.custom.ctpRed.opacity(0.1))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.custom.ctpRed.opacity(0.2), lineWidth: 1)
                                )
                                .cornerRadius(8)
                        }

                        ToraButton(
                            "sign in",
                            size: .medium,
                            backgroundColor: Color.custom.ctpBlue.opacity(0.20),
                            borderColor: Color.custom.ctpBlue.opacity(0.60),
                            textColor: Color.custom.ctpBlue,
                            cornerRadius: 8,
                            fullWidth: true,
                            isLoading: isLoading,
                            loadingTitle: "signing in...",
                            systemImage: isLoading ? nil : "arrow.right",
                            action: { signIn() }
                        )
                        .disabled(isLoading || email.isEmpty || password.isEmpty)
                    }
                    .padding(16)
                }
                .frame(maxWidth: 480)
                .background(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.custom.ctpSurface0.opacity(0.3), lineWidth: 1)
                )
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .padding(.horizontal, 16)

                Spacer(minLength: 16)
            }
        }
    }

    // MARK: - Private Methods

    private func signIn() {
        isLoading = true
        errorMessage = ""

        Task {
            defer {
                isLoading = false
            }

            do {
                await authService.login(
                    email: email,
                    password: password
                )
                dismiss()
            }
        }
    }
}

#Preview {
    LoginFormSheet()
        .environmentObject(AuthService())
}
