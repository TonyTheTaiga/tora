import Combine
import Foundation
import SwiftData
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
        case createdAt = "created_at"
        case role
    }
}

struct HyperParam: Codable, Equatable {
    let key: String
    let value: HyperParamValue

    enum HyperParamValue: Codable, Equatable {
        case string(String)
        case int(Int)
        case double(Double)
        case bool(Bool)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()

            if let stringValue = try? container.decode(String.self) {
                self = .string(stringValue)
            } else if let intValue = try? container.decode(Int.self) {
                self = .int(intValue)
            } else if let doubleValue = try? container.decode(Double.self) {
                self = .double(doubleValue)
            } else if let boolValue = try? container.decode(Bool.self) {
                self = .bool(boolValue)
            } else {
                throw DecodingError.typeMismatch(
                    HyperParamValue.self,
                    DecodingError.Context(
                        codingPath: decoder.codingPath,
                        debugDescription: "Unsupported hyperparam value type"
                    )
                )
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .string(let value):
                try container.encode(value)
            case .int(let value):
                try container.encode(value)
            case .double(let value):
                try container.encode(value)
            case .bool(let value):
                try container.encode(value)
            }
        }

        var displayValue: String {
            switch self {
            case .string(let value):
                return value
            case .int(let value):
                return String(value)
            case .double(let value):
                return String(value)
            case .bool(let value):
                return String(value)
            }
        }
    }
}

struct Experiment: Decodable, Identifiable, Equatable {
    var id: String
    var name: String
    var description: String?
    var hyperparams: [HyperParam]
    var tags: [String]
    var createdAt: Date
    var updatedAt: Date
    var availableMetrics: [String]
    var workspaceId: String?
    var url: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case hyperparams
        case tags
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case availableMetrics = "available_metrics"
        case workspaceId = "workspace_id"
        case url
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

            guard let token = authService.currentUser?.authToken else {
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
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode)
                throw WorkspaceErrors.requestError(httpResponse.statusCode, errorMessage)
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(ApiResponse<[Workspace]>.self, from: data)
                self.workspaces = apiResponse.data ?? []
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }

    public func listExperiments(for workspaceId: String) async throws -> [Experiment] {
        try await measure(OSLog.workspace, name: "listExperiments") {
            guard let url = URL(string: "\(baseUrl)/workspaces/\(workspaceId)/experiments") else {
                throw WorkspaceErrors.invalidURL
            }

            guard let token = authService.currentUser?.authToken else {
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
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode)
                throw WorkspaceErrors.requestError(httpResponse.statusCode, errorMessage)
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(ApiResponse<[Experiment]>.self, from: data)
                return apiResponse.data ?? []
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }
}

// MARK: - Experiment Service

@MainActor
class ExperimentService: ObservableObject {
    // MARK: - Properties

    @Published var experiments: [Experiment] = []
    private var baseUrl = Config.baseURL + "/api"
    private let authService: AuthService
    static let shared: ExperimentService = .init(authService: AuthService.shared)

    // MARK: - Constructor

    init(authService: AuthService) {
        self.authService = authService
    }

    // MARK: - Public Methods

    public func listAll() async throws {
        try await measure(OSLog.workspace, name: "listAllExperiments") {
            guard let url = URL(string: "\(baseUrl)/experiments") else {
                throw WorkspaceErrors.invalidURL
            }

            guard let token = authService.currentUser?.authToken else {
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
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode)
                throw WorkspaceErrors.requestError(httpResponse.statusCode, errorMessage)
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(ApiResponse<[Experiment]>.self, from: data)
                self.experiments = apiResponse.data ?? []
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }

    public func get(experimentId: String) async throws -> Experiment {
        try await measure(OSLog.workspace, name: "getExperiment") {
            guard let url = URL(string: "\(baseUrl)/experiments/\(experimentId)") else {
                throw WorkspaceErrors.invalidURL
            }

            guard let token = authService.currentUser?.authToken else {
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
                let errorMessage = HTTPURLResponse.localizedString(
                    forStatusCode: httpResponse.statusCode)
                throw WorkspaceErrors.requestError(httpResponse.statusCode, errorMessage)
            }

            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let apiResponse = try decoder.decode(ApiResponse<Experiment>.self, from: data)
                guard let experiment = apiResponse.data else {
                    throw WorkspaceErrors.dataError("No experiment data received")
                }
                return experiment
            } catch {
                throw WorkspaceErrors.jsonParsingError(error)
            }
        }
    }
}
