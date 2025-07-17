import Combine
import SwiftData
import SwiftUI

struct LoginFormSheet: View {
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

                Button(action: {
                    Task { await signIn() }
                }) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isLoading ? "Signing In..." : "Sign In")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isLoading || email.isEmpty || password.isEmpty)

                Spacer()
            }
            .padding()
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func signIn() async {
        isLoading = true
        showError = false
        errorMessage = ""

        defer {
            isLoading = false
        }

        do {
            let userSession = try await authService.login(email: email, password: password)
            context.insert(userSession)
            try context.save()
            dismiss()
        } catch {
            if let authError = error as? LocalizedError {
                errorMessage = authError.errorDescription ?? "Authentication failed"
            } else {
                errorMessage = "An unexpected error occurred. Please try again."
            }
            showError = true
        }
    }
}
