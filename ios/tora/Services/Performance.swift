import Foundation
import os

func measure<T>(
    _ logger: Logger,
    name: String,
    work: () async throws -> T
) async throws -> T {
    let startTime = DispatchTime.now()
    do {
        let result = try await work()
        let endTime = DispatchTime.now()
        let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000_000

        logger.debug("\(name) finished in \(timeInterval, format: .fixed(precision: 3))s")
        return result
    } catch {
        let endTime = DispatchTime.now()
        let nanoTime = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000_000

        logger.error(
            "\(name) failed after \(timeInterval, format: .fixed(precision: 3))s with error: \(error.localizedDescription)"
        )
        throw error
    }
}

extension Double {
    func formatted(_ f: String) -> String {
        String(format: "%\(f)f", self)
    }
}

extension OSLog {
    private static var subsystem = Bundle.main.bundleIdentifier!
    public static let auth = Logger(subsystem: subsystem, category: "auth")
    public static let workspace = Logger(subsystem: subsystem, category: "workspace")
}
