import Combine
import SwiftData
import SwiftUI

struct SignInSheet: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.modelContext) private var context
    @StateObject private var authService = AuthService()
    @State private var email = ""
    @State private var password = ""
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
                    .keyboardType(.emailAddress)

                SecureField("Password", text: $password)
                    .focused($isFocused, equals: .password)
                    .onSubmit {
                        isFocused = .email
                    }
                    .textFieldStyle(RoundedBorderTextFieldStyle())

                Button("Sign In") {
                    Task {
                        do {
                            try await signIn()
                        } catch {
                            print("failed to login")
                        }

                    }
                }
                .buttonStyle(.borderedProminent)

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
        .modalBackground()
    }

    private func signIn() async throws {
        do {
            let userSession = try await authService.login(email: email, password: password)
            print(userSession.auth_token)
            context.insert(userSession)
            dismiss()
        } catch {
            throw AuthErrors.authFailure
        }
    }
}
