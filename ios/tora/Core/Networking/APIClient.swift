import Foundation

struct APIClient {
    let baseURL: String
    let authService: AuthService

    func get(path: String) async throws -> Data {
        guard let url = URL(string: baseURL + path) else {
            throw ServiceError.invalidURL
        }

        let token = try await authService.getAuthToken()
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw ServiceError.invalidResponse
            }

            guard (200...299).contains(httpResponse.statusCode) else {
                let message = HTTPURLResponse.localizedString(forStatusCode: httpResponse.statusCode)
                throw ServiceError.requestError(httpResponse.statusCode, message)
            }
            return data
        } catch let error as ServiceError {
            throw error
        } catch {
            throw ServiceError.networkError(error)
        }
    }
}
