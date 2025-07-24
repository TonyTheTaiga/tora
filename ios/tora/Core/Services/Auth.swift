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

// MARK: - User Session Model

class UserSession: Encodable, Decodable {
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

// MARK: - Keychain Errors

enum KeychainError: Error {
    case noPassword
    case unexpectedPasswordData
    case unhandledError(status: OSStatus)
    case invalidData
}

// MARK: - Authentication Service

@MainActor
class AuthService: ObservableObject {
    // MARK: - Properties

    @Published var isAuthenticated = false
    @Published var currentUser: UserSession?

    private let serviceName = "tora-tracker"
    private let backendUrl: String = Config.baseURL
    static let shared: AuthService = .init()

    // MARK: - Constructor

    private init() {
        checkAuthenticationStatus()
    }

    // MARK: - Public Methods

    func logout() {
        isAuthenticated = false
        currentUser = nil
    }

    func login(email: String, password: String) async throws {
        do {
            let userSession = try await loginWithEmailAndPassword(
                email: email,
                password: password
            )
            try storeSessionInKeychain(userSession)
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

    private func checkAuthenticationStatus() {
        isAuthenticated = false
        currentUser = nil

        if checkSessionInKeychain() {
            // still need to verify it the auth token is valid here.
            currentUser = try? retrieveSessionFromKeychain()
            isAuthenticated = true
        }
    }

    private func jsonSerialize(_ userSession: UserSession) throws -> Data {
        return try JSONEncoder().encode(userSession)
    }

    private func jsonDeserialize(_ input: Data) throws -> UserSession {
        return try JSONDecoder().decode(UserSession.self, from: input)
    }

    private func storeSessionInKeychain(_ userSession: UserSession) throws {
        let serialized = try jsonSerialize(userSession)

        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: userSession.email,
            kSecAttrService as String: serviceName,
            kSecValueData as String: serialized,
        ]
        var status = SecItemAdd(query as CFDictionary, nil)

        // If duplicate just update the key
        if status == errSecDuplicateItem {
            query.removeValue(forKey: kSecValueData as String)
            let attrs: [String: Any] = [
                kSecValueData as String: serialized
            ]
            status = SecItemUpdate(
                query as CFDictionary,
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

    private func checkSessionInKeychain() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        return status == errSecSuccess
    }

    private func retrieveSessionFromKeychain() throws -> UserSession {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess else {
            throw KeychainError.unhandledError(status: status)
        }
        guard let userSessionData = result as? Data else {
            throw KeychainError.invalidData
        }
        return try jsonDeserialize(userSessionData)
    }

    private func loginWithEmailAndPassword(email: String, password: String)
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
