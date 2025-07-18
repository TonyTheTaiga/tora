//
//  Workspaces.swift
//  tora
//
//  Created by taiga on 7/18/25.
//
import Combine
import Foundation
import SwiftData
import SwiftUI

struct ApiResponse<T: Decodable>: Decodable {
    let status: Int
    let data: T?
}

enum WorkspaceErrors: Error, LocalizedError {
    case invalidURL
    case authFailure(String)
    case dataError(String)
    case responseError(String)
    case requestError(Int, String)
    case networkError(Error)
    case jsonParsingError(Error)
    case invalidResponse
    case missingRequiredFields([String])
    case clientError(String)
    case unauthenticated

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
        case .clientError(let message):
            return "Client error \(message)"
        case .unauthenticated:
            return "User is not authenticated."
        }
    }
}

struct Workspace: Decodable, Identifiable {
    var id: String
    var name: String
    var description: String?
    var createdAt: Date
    var role: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case createdAt = "created_at"
        case role
    }
}

@MainActor
class WorkspaceService: ObservableObject {
    private var baseUrl = "http://localhost:8080/api"
    private let authService: AuthService

    init(authService: AuthService) {
        self.authService = authService
    }

    public func list() async throws -> [Workspace] {
        guard let url = URL(string: "\(baseUrl)/workspaces") else {
            throw WorkspaceErrors.invalidURL
        }

        guard let token = authService.currentUser?.auth_token else {
            throw WorkspaceErrors.unauthenticated
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw WorkspaceErrors.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw WorkspaceErrors.invalidResponse
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let errorMessage = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
            throw WorkspaceErrors.requestError(httpResponse.statusCode, errorMessage)
        }

        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let apiResponse = try decoder.decode(ApiResponse<[Workspace]>.self, from: data)
            return apiResponse.data ?? []
        } catch {
            throw WorkspaceErrors.jsonParsingError(error)
        }
    }
}
