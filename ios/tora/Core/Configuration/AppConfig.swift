import Foundation

enum Config {
    static let baseURL: String = {
        let process = ProcessInfo.processInfo

        if let trimmed = process.environment["API_BASE_URL"]?.trimmingCharacters(in: .whitespacesAndNewlines),
            !trimmed.isEmpty
        {
            return trimmed
        }

        fatalError("API_BASE_URL is not configured. Set it in the environment before launching.")
    }()
}
