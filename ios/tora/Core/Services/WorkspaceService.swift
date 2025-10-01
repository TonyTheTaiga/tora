import Combine
import Foundation
import os

@MainActor
final class WorkspaceService: ObservableObject {
    @Published private(set) var workspaces: [Workspace] = []

    private let client: APIClient

    init(authService: AuthService, baseURL: String) {
        self.client = APIClient(baseURL: baseURL, authService: authService)
    }

    convenience init(authService: AuthService) {
        self.init(authService: authService, baseURL: Config.baseURL)
    }

    func list() async throws {
        try await measure(OSLog.workspace, name: "listWorkspaces") {
            let data = try await client.get(path: "/workspaces")
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let response = try decoder.decode(APIResponse<[Workspace]>.self, from: data)
                self.workspaces = response.data ?? []
            } catch {
                throw ServiceError.jsonParsingError(error)
            }
        }
    }

    func listExperiments(for workspaceId: String) async throws -> [Experiment] {
        try await measure(OSLog.workspace, name: "listExperiments") {
            let data = try await client.get(path: "/workspaces/\(workspaceId)/experiments")
            do {
                let decoder = JSONDecoder()
                decoder.dateDecodingStrategy = .iso8601
                let response = try decoder.decode(APIResponse<[Experiment]>.self, from: data)
                return response.data ?? []
            } catch {
                throw ServiceError.jsonParsingError(error)
            }
        }
    }
}
