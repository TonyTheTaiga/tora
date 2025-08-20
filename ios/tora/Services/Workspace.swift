import Combine
import Foundation
import SwiftUI
import os

// MARK: - API Response

struct ApiResponse<T: Decodable>: Decodable {
    let status: Int
    let data: T?
}

// MARK: - Workspace Errors

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

// MARK: - Data Models

struct Workspace: Decodable, Identifiable, Equatable {
    var id: String
    var name: String
    var description: String?
    var createdAt: Date
    var role: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case createdAt
        case role
    }
}

// MARK: - Workspace Service

@MainActor
class WorkspaceService: ObservableObject {
    // MARK: - Properties

    @Published var workspaces: [Workspace] = []
    private var baseUrl = Config.baseURL + "/api"
    private let authService: AuthService
    static let shared: WorkspaceService = .init(authService: AuthService.shared)

    // MARK: - Constructor

    init(authService: AuthService) {
        self.authService = authService
    }

    // MARK: - Public Methods

    public func list() async throws {
        try await measure(OSLog.workspace, name: "listWorkspaces") {
            guard let url = URL(string: "\(baseUrl)/workspaces") else {
                throw WorkspaceErrors.invalidURL
            }

            let token = try await authService.getAuthToken()
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue(
                "application/json",
                forHTTPHeaderField: "Content-Type"
            )
            request.setValue(
                "Bearer \(token)",
                forHTTPHeaderField: "Authorization"
            )

            let (data, response): (Data, URLResponse)
            do {
                (data, response) = try await URLSession.shared.data(
                    for: request
                )
            } catch {
                throw WorkspaceErrors.networkError(error)
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw WorkspaceErrors.invalidResponse
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode
                )
                throw WorkspaceErrors.requestError(
                    httpResponse.statusCode,
                    errorMessage
                )
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(
                    ApiResponse<[Workspace]>.self,
                    from: data
                )
                self.workspaces = apiResponse.data ?? []
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }

    public func listExperiments(for workspaceId: String) async throws
        -> [Experiment]
    {
        try await measure(OSLog.workspace, name: "listExperiments") {
            guard
                let url = URL(
                    string: "\(baseUrl)/workspaces/\(workspaceId)/experiments"
                )
            else {
                throw WorkspaceErrors.invalidURL
            }

            let token = try await authService.getAuthToken()
            var request = URLRequest(url: url)
            request.httpMethod = "GET"
            request.setValue(
                "application/json",
                forHTTPHeaderField: "Content-Type"
            )
            request.setValue(
                "Bearer \(token)",
                forHTTPHeaderField: "Authorization"
            )

            let (data, response): (Data, URLResponse)
            do {
                (data, response) = try await URLSession.shared.data(
                    for: request
                )
            } catch {
                throw WorkspaceErrors.networkError(error)
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                throw WorkspaceErrors.invalidResponse
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode
                )
                throw WorkspaceErrors.requestError(
                    httpResponse.statusCode,
                    errorMessage
                )
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(
                    ApiResponse<[Experiment]>.self,
                    from: data
                )
                return apiResponse.data ?? []
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }
}
