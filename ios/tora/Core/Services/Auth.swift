import Combine
import Foundation
import SwiftData
import SwiftUI
import os

// MARK: - Private Data Structures

struct Credentials {
    var email: String
    var password: String
}

private struct UserData: Decodable {
    let id: String
    let email: String
}

private struct TokenData: Decodable {
    let user: UserData
    let accessToken: String
    let refreshToken: String
    let tokenType: String
    let expiresIn: Int
    let expiresAt: Int
}

private struct LoginResponse: Decodable {
    let data: TokenData
}

// MARK: - Authentication Errors

enum AuthErrors: Error, LocalizedError {
    case invalidURL
    case authFailure(String)
    case dataError(String)
    case responseError(String)
    case requestError(Int, String)
    case networkError(Error)
    case jsonParsingError(Error)
    case invalidResponse
    case missingRequiredFields([String])

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL configuration"
        case .authFailure(let message):
            return "Authentication failed: \(message)"
        case .dataError(let message):
            return "Data processing error: \(message)"
        case .responseError(let message):
            return "Response error: \(message)"
        case .requestError(let statusCode, let message):
            return "Request failed with status \(statusCode): \(message)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .jsonParsingError(let error):
            return "JSON parsing error: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response format"
        case .missingRequiredFields(let fields):
            return "Missing required fields: \(fields.joined(separator: ", "))"
        }
    }
}

enum KeychainError: Error {
    case noPassword
    case unexpectedPasswordData
    case unhandledError(status: OSStatus)
}

// MARK: - User Session Model

@Model
class UserSession {
    var id: String
    var email: String
    var authToken: String
    var refreshToken: String
    var expiresIn: Date
    var expiresAt: Date
    var tokenType: String

    init(
        id: String,
        email: String,
        authToken: String,
        refreshToken: String,
        expiresIn: Date,
        expiresAt: Date,
        tokenType: String
    ) {
        self.id = id
        self.email = email
        self.authToken = authToken
        self.refreshToken = refreshToken
        self.expiresAt = expiresAt
        self.expiresIn = expiresIn
        self.tokenType = tokenType
    }
}

// MARK: - Authentication Service

@MainActor
class AuthService: ObservableObject {
    // MARK: - Properties

    @Published var isAuthenticated = false
    @Published var currentUser: UserSession?
    @Environment(\.modelContext) private var modelContext

    private let backendUrl: String = Config.baseURL
    static let shared: AuthService = .init()

    // MARK: - Constructor

    private init() {
        checkAuthenticationStatus()
    }

    // MARK: - Public Methods

    func checkAuthenticationStatus() {
        // update this to fetch from keychain then do the auth? or is it better to store the tokens in the keychain?
        isAuthenticated = false
        currentUser = nil
    }

    func logout() {
        isAuthenticated = false
        currentUser = nil
    }

    func login(email: String, password: String) async throws {
        do {
            let userSession = try await _login_with_email_and_password(
                email: email,
                password: password
            )
            try updateKeychain(email: email, password: password)
            self.isAuthenticated = true
            self.currentUser = userSession
        } catch let authError as AuthErrors {
            throw authError
        } catch let keychainError as KeychainError {
            throw keychainError
        } catch {
            throw AuthErrors.authFailure(
                "Unexpected error: \(error.localizedDescription)"
            )
        }
    }

    // MARK: - Private Methods

    private func updateKeychain(email: String, password: String) throws {
        // add -> catch -> update
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassInternetPassword,
            kSecAttrAccount as String: email,
            kSecAttrServer as String: "tora-tracker",
            kSecValueData as String: password.data(
                using: String.Encoding.utf8
            )!,
        ]
        var status = SecItemAdd(addQuery as CFDictionary, nil)
        if status == errSecDuplicateItem {
            let updateQuery: [String: Any] = [
                kSecClass as String: kSecClassInternetPassword,
                kSecAttrAccount as String: email,
                kSecAttrServer as String: "tora-tracker",
            ]
            let attrs: [String: Any] = [
                kSecValueData as String: password.data(
                    using: String.Encoding.utf8
                )!
            ]
            status = SecItemUpdate(
                updateQuery as CFDictionary,
                attrs as CFDictionary
            )
        }

        if status != errSecSuccess {
            let errorMessage =
                SecCopyErrorMessageString(status, nil) as String?
                ?? "Unknown error"
            print("updating keychain failed: \(errorMessage) (\(status))")
            throw KeychainError.unhandledError(status: status)
        }
    }

    private func _login_with_email_and_password(email: String, password: String)
        async throws -> UserSession
    {
        try await measure(OSLog.auth, name: "_login_with_email_and_password") {
            guard let url = URL(string: "\(backendUrl)/api/login") else {
                throw AuthErrors.invalidURL
            }

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue(
                "application/json",
                forHTTPHeaderField: "Content-Type"
            )

            do {
                request.httpBody = try JSONSerialization.data(withJSONObject: [
                    "email": email, "password": password,
                ])
            } catch {
                throw AuthErrors.dataError(
                    "Failed to serialize login credentials: \(error.localizedDescription)"
                )
            }

            let (data, response): (Data, URLResponse)
            do {
                (data, response) = try await URLSession.shared.data(
                    for: request
                )
            } catch {
                throw AuthErrors.networkError(error)
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw AuthErrors.invalidResponse
            }

            guard httpResponse.statusCode == 200 else {
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode
                )
                throw AuthErrors.requestError(
                    httpResponse.statusCode,
                    errorMessage
                )
            }

            do {
                let decoder = JSONDecoder()
                decoder.keyDecodingStrategy = .convertFromSnakeCase
                decoder.dateDecodingStrategy = .iso8601
                let loginResponse = try decoder.decode(
                    LoginResponse.self,
                    from: data
                )
                let tokenData = loginResponse.data
                let expiresInDate = Date(
                    timeIntervalSince1970: TimeInterval(tokenData.expiresIn)
                )
                let expiresAtDate = Date(
                    timeIntervalSince1970: TimeInterval(tokenData.expiresAt)
                )
                let session = UserSession(
                    id: tokenData.user.id,
                    email: tokenData.user.email,
                    authToken: tokenData.accessToken,
                    refreshToken: tokenData.refreshToken,
                    expiresIn: expiresInDate,
                    expiresAt: expiresAtDate,
                    tokenType: tokenData.tokenType
                )

                return session
            } catch {
                throw AuthErrors.jsonParsingError(error)
            }
        }
    }
}
