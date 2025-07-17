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

enum AuthErrors: Error {
    case invalidURL
    case authFailure
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

class AuthService: ObservableObject {
    @Published var isAuthenticated = false

    private let backendUrl: String = "http://localhost:8080"

    func login(email: String, password: String) async throws -> UserSession {
        guard let url = URL(string: "\(backendUrl)/api/login") else {
            throw AuthErrors.invalidURL
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["email": email, "password": password])
        let (data, response) = try await Foundation.URLSession.shared.data(for: request)
        let httpResponse = response as! HTTPURLResponse

        guard httpResponse.statusCode == 200 else {
            throw AuthErrors.authFailure
        }

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        guard let responseData = json["data"] as? [String: Any] else {
            throw AuthErrors.authFailure
        }

        guard let userData = responseData["user"] as? [String: Any],
            let userId = userData["id"] as? String,
            let userEmail = userData["email"] as? String
        else {
            throw AuthErrors.authFailure
        }

        guard let accessToken = responseData["access_token"] as? String,
            let refreshToken = responseData["refresh_token"] as? String,
            let tokenType = responseData["token_type"] as? String,
            let expiresIn = responseData["expires_in"] as? Int,
            let expiresAt = responseData["expires_at"] as? Int
        else {
            throw AuthErrors.authFailure
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

        isAuthenticated = true
        return session
    }
}
