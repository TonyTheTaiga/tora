import Combine
import Foundation

@MainActor
final class ExperimentDetailViewModel: ObservableObject {
    struct ResultItem: Identifiable {
        let id = UUID()
        let name: String
        let value: Double
    }

    @Published private(set) var metricsByName: [String: [Metric]] = [:]
    @Published private(set) var results: [ResultItem] = []

    func load(experimentID: String, service: ExperimentService) async {
        do {
            let resultRows = try await service.getResults(experimentId: experimentID)
            let metricRows = try await service.getMetrics(experimentId: experimentID)
            let parsed = Self.parseLogs(resultRows + metricRows)
            results = parsed.results
            metricsByName = parsed.metricsByName
        } catch {
            // keep existing values on failures
        }
    }

    private static func parseLogs(_ rows: [Metric]) -> (results: [ResultItem], metricsByName: [String: [Metric]]) {
        let resultRows = rows.filter { $0.metadata?.type?.lowercased() == "result" }
        let groupedResults = Dictionary(grouping: resultRows, by: { $0.name })
        let results: [ResultItem] = groupedResults.compactMap { name, list in
            guard let value = list.first?.value else { return nil }
            return ResultItem(name: name, value: value)
        }
        .sorted { $0.name < $1.name }

        let metricRows = rows.filter { $0.metadata?.type?.lowercased() == "metric" }
        var groupedMetrics = Dictionary(grouping: metricRows, by: { $0.name })
        for (key, series) in groupedMetrics {
            groupedMetrics[key] = series.sorted { a, b in
                let sa = a.step ?? .min
                let sb = b.step ?? .min
                if sa == sb {
                    return a.createdAt < b.createdAt
                }
                return sa < sb
            }
        }

        return (results, groupedMetrics)
    }
}
