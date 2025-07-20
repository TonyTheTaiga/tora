import Combine
import Foundation
import SwiftData
import SwiftUI

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
        id: String, email: String, authToken: String, refreshToken: String, expiresIn: Date,
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

@MainActor
class AuthService: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: UserSession?

    // Magical singleton, gets instantited on first reference
    static let shared = AuthService()

    private let backendUrl: String = Config.baseURL

    private init() {
        checkAuthenticationStatus()
    }

    func checkAuthenticationStatus() {
        isAuthenticated = false
        currentUser = nil
    }

    func logout() {
        isAuthenticated = false
        currentUser = nil
    }

    func login(email: String, password: String) async throws -> UserSession {
        do {
            let session = try await doLogin(email: email, password: password)
            self.isAuthenticated = true
            self.currentUser = session
            return session
        } catch let authError as AuthErrors {
            throw authError
        } catch {
            throw AuthErrors.authFailure("Unexpected error: \(error.localizedDescription)")
        }
    }

    private func doLogin(email: String, password: String) async throws -> UserSession {
        guard let url = URL(string: "\(backendUrl)/api/login") else {
            throw AuthErrors.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "email": email, "password": password,
            ])
        } catch {
            throw AuthErrors.dataError(
                "Failed to serialize login credentials: \(error.localizedDescription)")
        }

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthErrors.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw AuthErrors.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            let errorMessage = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            throw AuthErrors.requestError(httpResponse.statusCode, errorMessage)
        }

        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .iso8601
            let loginResponse = try decoder.decode(LoginResponse.self, from: data)
            let tokenData = loginResponse.data

            let expiresInDate = Date(timeIntervalSince1970: TimeInterval(tokenData.expiresIn))
            let expiresAtDate = Date(timeIntervalSince1970: TimeInterval(tokenData.expiresAt))

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
