import Combine
import Foundation
import os

@MainActor
final class ExperimentService: ObservableObject {
    @Published private(set) var experiments: [Experiment] = []

    private let client: APIClient

    init(authService: AuthService, baseURL: String) {
        self.client = APIClient(baseURL: baseURL, authService: authService)
    }

    convenience init(authService: AuthService) {
        self.init(authService: authService, baseURL: Config.baseURL)
    }

    func listAll() async throws {
        try await measure(OSLog.workspace, name: "listAllExperiments") {
            let data = try await client.get(path: "/experiments")
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let response = try decoder.decode(APIResponse<[Experiment]>.self, from: data)
                self.experiments = response.data ?? []
            } catch {
                throw ServiceError.jsonParsingError(error)
            }
        }
    }

    func get(experimentId: String) async throws -> Experiment {
        try await measure(OSLog.workspace, name: "getExperiment") {
            let data = try await client.get(path: "/experiments/\(experimentId)")
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let response = try decoder.decode(APIResponse<Experiment>.self, from: data)
                guard let experiment = response.data else {
                    throw ServiceError.dataError("No experiment data received")
                }
                return experiment
            } catch {
                throw ServiceError.jsonParsingError(error)
            }
        }
    }

    func getLogs(experimentId: String) async throws -> [Metric] {
        try await measure(OSLog.workspace, name: "getExperimentLogs") {
            let data = try await client.get(path: "/experiments/\(experimentId)/logs")
            return try decodeMetrics(from: data)
        }
    }

    func getMetrics(experimentId: String) async throws -> [Metric] {
        try await measure(OSLog.workspace, name: "getMetrics") {
            let data = try await client.get(path: "/experiments/\(experimentId)/metrics")
            return try decodeMetrics(from: data)
        }
    }

    func getResults(experimentId: String) async throws -> [Metric] {
        try await measure(OSLog.workspace, name: "getResults") {
            let data = try await client.get(path: "/experiments/\(experimentId)/results")
            return try decodeMetrics(from: data)
        }
    }

    private func decodeMetrics(from data: Data) throws -> [Metric] {
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let response = try decoder.decode(APIResponse<[Metric]>.self, from: data)
            return response.data ?? []
        } catch {
            throw ServiceError.jsonParsingError(error)
        }
    }
}
