import SwiftUI
import UIKit

// MARK: - Experiment Detail View

struct ExperimentsView: View {
    // MARK: - Properties

    @Binding var experiment: Experiment
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject private var experimentService: ExperimentService

    // MARK: - Body

    var body: some View {
        ExperimentContentView(experiment: experiment) {
            await refreshExperiment()
        }
        .navigationTitle("")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text(experiment.name)
                    .font(.headline)
                    .multilineTextAlignment(.center)
                    .lineLimit(horizontalSizeClass == .regular ? 2 : 1)
                    .truncationMode(horizontalSizeClass == .regular ? .tail : .middle)
                    .minimumScaleFactor(0.9)
            }
        }
    }

    private func refreshExperiment() async {
        do {
            let updated = try await experimentService.get(experimentId: experiment.id)
            await MainActor.run { self.experiment = updated }
        } catch {
            // ignore errors on pull-to-refresh; keep last value
        }
    }
}

// MARK: - Experiment Content View

struct ExperimentContentView: View {
    // MARK: - Properties

    let experiment: Experiment
    let onRefresh: () async -> Void
    @EnvironmentObject private var experimentService: ExperimentService
    @State private var results: [(name: String, value: Double)] = []
    @State private var metricsByName: [String: [Metric]] = [:]

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Description
                if let description = experiment.description, !description.isEmpty {
                    Text(description)
                        .font(.body)
                        .foregroundStyle(Color.custom.ctpSubtext0)
                }

                // Tags
                if !experiment.tags.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Text("Tags")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Color.custom.ctpText)
                            Text("[\(experiment.tags.count)]")
                                .font(.subheadline)
                                .foregroundStyle(Color.custom.ctpSubtext0)
                        }
                        // Wrap tags using adaptive grid
                        LazyVGrid(
                            columns: [
                                GridItem(.adaptive(minimum: 48), spacing: 2, alignment: .leading)
                            ],
                            alignment: .leading,
                            spacing: 2
                        ) {
                            ForEach(experiment.tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption2)
                                    .lineLimit(1)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.custom.ctpBlue.opacity(0.20))
                                    .foregroundStyle(Color.custom.ctpBlue)
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(
                                                Color.custom.ctpBlue.opacity(0.40), lineWidth: 1)
                                    )
                                    .clipShape(RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                }

                // Hyperparameters
                if !experiment.hyperparams.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Text("Hyperparameters")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Color.custom.ctpText)
                            Text("[\(experiment.hyperparams.count)]")
                                .font(.subheadline)
                                .foregroundStyle(Color.custom.ctpSubtext0)
                        }
                        VStack(spacing: 0) {
                            ForEach(Array(experiment.hyperparams.enumerated()), id: \.offset) {
                                index, param in
                                HStack(alignment: .center) {
                                    Text(param.key)
                                        .font(.caption)
                                        .foregroundStyle(Color.custom.ctpSubtext0)
                                        .lineLimit(1)
                                    Spacer(minLength: 8)
                                    Text(param.value.displayValue)
                                        .font(.caption)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                        .background(Color.custom.ctpSurface0.opacity(0.20))
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6)
                                                .stroke(
                                                    Color.custom.ctpSurface0.opacity(0.30),
                                                    lineWidth: 1)
                                        )
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 8)
                                .background(
                                    index % 2 == 0
                                        ? Color.custom.ctpSurface0.opacity(0.06) : Color.clear)
                                if index != experiment.hyperparams.count - 1 {
                                    Divider()
                                        .overlay(Color.custom.ctpSurface0.opacity(0.20))
                                }
                            }
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.custom.ctpMantle)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.custom.ctpSurface0.opacity(0.30), lineWidth: 1)
                        )
                    }
                }

                // Results
                if !results.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack(spacing: 6) {
                            Text("Results")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(Color.custom.ctpText)
                            Text("[\(results.count)]")
                                .font(.subheadline)
                                .foregroundStyle(Color.custom.ctpSubtext0)
                        }
                        VStack(spacing: 0) {
                            ForEach(Array(results.enumerated()), id: \.offset) { index, item in
                                HStack {
                                    Circle().fill(Color.custom.ctpGreen).frame(width: 6, height: 6)
                                    Text(item.name)
                                        .font(.caption)
                                        .foregroundStyle(Color.custom.ctpText)
                                        .lineLimit(1)
                                    Spacer()
                                    Text(String(format: "%.4f", item.value))
                                        .font(.caption)
                                        .foregroundStyle(Color.custom.ctpBlue)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 8)
                                .background(
                                    index % 2 == 0
                                        ? Color.custom.ctpSurface0.opacity(0.06) : Color.clear)
                                if index != results.count - 1 {
                                    Divider()
                                        .overlay(Color.custom.ctpSurface0.opacity(0.20))
                                }
                            }
                        }
                        .background(
                            RoundedRectangle(cornerRadius: 8).fill(Color.custom.ctpMantle)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 8).stroke(
                                Color.custom.ctpSurface0.opacity(0.30), lineWidth: 1)
                        )
                    }
                }

                // Metadata
                VStack(alignment: .leading, spacing: 4) {
                    Text("Metadata")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Color.custom.ctpText)
                    HStack {
                        Text("Created:")
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                        Text(dateString(experiment.createdAt))
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpText)
                    }
                    HStack {
                        Text("Updated:")
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                        Text(dateString(experiment.updatedAt))
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpText)
                    }
                    HStack {
                        Text("Experiment ID:")
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                        Text(experiment.id)
                            .font(.caption)
                            .foregroundStyle(Color.custom.ctpText)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    if let workspaceId = experiment.workspaceId {
                        HStack {
                            Text("Workspace ID:")
                                .font(.caption)
                                .foregroundStyle(Color.custom.ctpSubtext0)
                            Text(workspaceId)
                                .font(.caption)
                                .foregroundStyle(Color.custom.ctpText)
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }
                    }
                }
            }
            .padding()
        }
        .task { await loadMetrics() }
        .refreshable {
            await onRefresh()
            await loadMetrics(force: true)
        }
    }
}

// MARK: - Helpers

private func dateString(_ date: Date) -> String {
    let fmt = DateFormatter()
    fmt.dateStyle = .medium
    fmt.timeStyle = .short
    return fmt.string(from: date)
}

extension ExperimentContentView {
    func loadMetrics(force: Bool = false) async {
        do {
            let rows = try await experimentService.getLogs(experimentId: experiment.id)
            await MainActor.run {
                // Results: metadata.type == "result"; assume single entry per key
                let resultRows = rows.filter { $0.metadata?.type?.lowercased() == "result" }
                let groupedResults = Dictionary(grouping: resultRows, by: { $0.name })
                self.results = groupedResults.compactMap { (name, list) -> (String, Double)? in
                    guard let value = list.first?.value else { return nil }
                    return (name, value)
                }.sorted { $0.0 < $1.0 }

                // Metrics for chart: metadata.type == "metric"; sort by step ascending within each series
                let metricRows = rows.filter { $0.metadata?.type?.lowercased() == "metric" }
                var groupedMetrics = Dictionary(grouping: metricRows, by: { $0.name })
                for (key, series) in groupedMetrics {
                    groupedMetrics[key] = series.sorted { (a, b) in
                        let sa = a.step ?? Int.min
                        let sb = b.step ?? Int.min
                        return sa < sb
                    }
                }
                self.metricsByName = groupedMetrics
            }
        } catch {
            // ignore errors
        }
    }
}

// MARK: - Preview

#Preview {
    @Previewable @State var sample = Experiment(
        id: "exp_123",
        name: "Sample Experiment",
        description: "A simple example experiment description.",
        hyperparams: [
            HyperParam(key: "learning_rate", value: .double(0.001)),
            HyperParam(key: "epochs", value: .int(12)),
            HyperParam(key: "use_dropout", value: .bool(true)),
            HyperParam(key: "optimizer", value: .string("adam")),
        ],
        tags: ["demo", "ios", "preview"],
        createdAt: Date(),
        updatedAt: Date(),
        workspaceId: nil,
        url: "https://example.com/exp_123"
    )

    return ExperimentsView(experiment: $sample)
}
