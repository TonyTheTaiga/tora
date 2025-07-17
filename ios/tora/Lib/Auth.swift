//
//  Auth.swift
//  tora
//
//  Created by taiga on 7/16/25.
//

import Combine
import Foundation
import SwiftData
import SwiftUI

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
    var auth_token: String
    var refresh_token: String
    var expiresIn: Date
    var expiresAt: Date
    var tokenType: String

    init(
        id: String, email: String, auth_token: String, refresh_token: String, expiresIn: Date, expiresAt: Date,
        tokenType: String
    ) {
        self.id = id
        self.email = email
        self.auth_token = auth_token
        self.refresh_token = refresh_token
        self.expiresAt = expiresAt
        self.expiresIn = expiresIn
        self.tokenType = tokenType
    }
}

@MainActor
class AuthService: ObservableObject {
    @Published var isAuthenticated = false
    @Published var currentUser: UserSession?

    static let shared = AuthService()
    private let backendUrl: String = "http://localhost:8080"

    private init() {
        checkAuthenticationStatus()
    }

    func checkAuthenticationStatus() {
        // Check if user session exists in SwiftData
        // This would typically check for stored session and validate token expiry
        // For now, we'll implement basic logic
        isAuthenticated = false
        currentUser = nil
    }

    func logout() {
        isAuthenticated = false
        currentUser = nil
        // Clear stored session data
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
            request.httpBody = try JSONSerialization.data(withJSONObject: ["email": email, "password": password])
        } catch {
            throw AuthErrors.dataError("Failed to serialize login credentials: \(error.localizedDescription)")
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

        let json: [String: Any]
        do {
            json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        } catch {
            throw AuthErrors.jsonParsingError(error)
        }

        guard let responseData = json["data"] as? [String: Any] else {
            throw AuthErrors.responseError("Missing 'data' field in response")
        }

        guard let userData = responseData["user"] as? [String: Any] else {
            throw AuthErrors.responseError("Missing 'user' field in response data")
        }

        guard let userId = userData["id"] as? String,
            let userEmail = userData["email"] as? String
        else {
            var missingFields: [String] = []
            if userData["id"] == nil { missingFields.append("user.id") }
            if userData["email"] == nil { missingFields.append("user.email") }
            throw AuthErrors.missingRequiredFields(missingFields)
        }

        guard let accessToken = responseData["access_token"] as? String,
            let refreshToken = responseData["refresh_token"] as? String,
            let tokenType = responseData["token_type"] as? String,
            let expiresIn = responseData["expires_in"] as? Int,
            let expiresAt = responseData["expires_at"] as? Int
        else {
            var missingFields: [String] = []
            if responseData["access_token"] == nil { missingFields.append("access_token") }
            if responseData["refresh_token"] == nil { missingFields.append("refresh_token") }
            if responseData["token_type"] == nil { missingFields.append("token_type") }
            if responseData["expires_in"] == nil { missingFields.append("expires_in") }
            if responseData["expires_at"] == nil { missingFields.append("expires_at") }
            throw AuthErrors.missingRequiredFields(missingFields)
        }

        let expiresInDate = Date(timeIntervalSince1970: TimeInterval(expiresIn))
        let expiresAtDate = Date(timeIntervalSince1970: TimeInterval(expiresAt))

        let session = UserSession(
            id: userId,
            email: userEmail,
            auth_token: accessToken,
            refresh_token: refreshToken,
            expiresIn: expiresInDate,
            expiresAt: expiresAtDate,
            tokenType: tokenType
        )

        return session
    }
}
