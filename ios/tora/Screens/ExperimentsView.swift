import Charts
import SwiftUI

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
            // Intentionally ignore on pull-to-refresh; preserve last good value
        }
    }
}

// MARK: - Experiment Content View

struct ExperimentContentView: View {
    // MARK: - Properties

    let experiment: Experiment
    let onRefresh: () async -> Void

    @EnvironmentObject private var experimentService: ExperimentService

    @State private var results: [ResultItem] = []
    @State private var metricsByName: [String: [Metric]] = [:]

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let description = experiment.description, !description.isEmpty {
                    Text(description)
                        .font(.body)
                        .foregroundStyle(Color.custom.ctpSubtext0)
                }

                if !metricsByName.isEmpty {
                    MetricsChartSectionView(series: metricsByName)
                }

                if !results.isEmpty {
                    ResultsSectionView(results: results)
                }

                if !experiment.hyperparams.isEmpty {
                    HyperparamsSectionView(hyperparams: experiment.hyperparams)
                }

                MetadataSectionView(
                    createdAt: experiment.createdAt,
                    updatedAt: experiment.updatedAt,
                    experimentId: experiment.id,
                    workspaceId: experiment.workspaceId,
                    tags: experiment.tags
                )
            }
            .padding()
        }
        .task { await loadMetrics() }
        .refreshable {
            await onRefresh()
            await loadMetrics()
        }
    }
}

// MARK: - Helpers

private typealias ResultItem = (name: String, value: Double)

extension Date {
    fileprivate var conciseString: String { formatted(date: .abbreviated, time: .shortened) }
}

extension ExperimentContentView {
    func loadMetrics() async {
        do {
            let logs = try await experimentService.getLogs(experimentId: experiment.id)
            await MainActor.run {
                let parsed = Self.parseLogs(logs)
                self.results = parsed.results
                self.metricsByName = parsed.metricsByName
            }
        } catch {
            // Ignore load errors; keep previous values
        }
    }

    fileprivate static func parseLogs(_ rows: [Metric]) -> (results: [ResultItem], metricsByName: [String: [Metric]]) {
        // Results: first value per unique name
        let resultRows = rows.filter { $0.metadata?.type?.lowercased() == "result" }
        let groupedResults = Dictionary(grouping: resultRows, by: { $0.name })
        let results: [ResultItem] =
            groupedResults
            .compactMap { name, list in
                guard let value = list.first?.value else { return nil }
                return (name, value)
            }
            .sorted { $0.name < $1.name }

        // Metrics: group by series name, sort by step
        let metricRows = rows.filter { $0.metadata?.type?.lowercased() == "metric" }
        var groupedMetrics = Dictionary(grouping: metricRows, by: { $0.name })
        for (key, series) in groupedMetrics {
            groupedMetrics[key] = series.sorted { (a, b) in
                let sa = a.step ?? .min
                let sb = b.step ?? .min
                return sa < sb
            }
        }

        return (results, groupedMetrics)
    }
}

// MARK: - Subviews

private struct HyperparamsSectionView: View {
    let hyperparams: [HyperParam]

    private var grid: [GridItem] { [GridItem(.adaptive(minimum: 140), spacing: 12, alignment: .leading)] }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Hyperparameters")

            LazyVGrid(columns: grid, alignment: .leading, spacing: 12) {
                ForEach(hyperparams, id: \.key) { param in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(param.key)
                            .font(.caption2)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                            .lineLimit(1)
                        Text(param.value.displayValue)
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(Color.custom.ctpText)
                            .lineLimit(1)
                    }
                }
            }
        }
    }
}

private struct ResultsSectionView: View {
    let results: [ResultItem]

    private var grid: [GridItem] { [GridItem(.adaptive(minimum: 140), spacing: 12, alignment: .leading)] }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            SectionHeader(title: "Results")
            LazyVGrid(columns: grid, alignment: .leading, spacing: 12) {
                ForEach(results, id: \.name) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.name)
                            .font(.caption2)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                            .lineLimit(1)
                        Text(item.value, format: .number.precision(.fractionLength(4)))
                            .font(.footnote.weight(.semibold))
                            .monospacedDigit()
                            .foregroundStyle(Color.custom.ctpText)
                            .lineLimit(1)
                    }
                }
            }
        }
    }
}

private struct MetricsChartSectionView: View {
    let series: [String: [Metric]]
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var yScaleMode: YScaleMode = .linear

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                SectionHeader(title: "Metrics")
                Spacer()
                Picker("Scale", selection: $yScaleMode) {
                    Text("Linear").tag(YScaleMode.linear)
                    Text("Log").tag(YScaleMode.log)
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 220)
            }

            Chart {
                LinePlot(
                    chartPoints,
                    x: .value("Step", \.step),
                    y: .value("Value", \.value),
                    series: .value("Metric", \.name)
                )
                .foregroundStyle(by: .value("Metric", \.name))
            }
            .chartLegend(.visible)
            .chartYAxis {
                AxisMarks(position: .leading) { value in
                    AxisGridLine()
                    AxisTick()
                    AxisValueLabel {
                        if let v = value.as(Double.self) {
                            Text(v, format: .number.precision(.fractionLength(2)))
                        }
                    }
                }
            }
            .applyYAxisScale(mode: effectiveScale, domain: yAxisDomain(for: effectiveScale))
            .frame(height: chartHeight)

            if yScaleMode == .log && hasNonPositiveValues {
                Text("Log scale requires positive values; showing linear scale.")
                    .font(.caption2)
                    .foregroundStyle(Color.custom.ctpSubtext0)
            }
        }
    }

    private var chartHeight: CGFloat {
        switch horizontalSizeClass {
        case .regular: return 360
        default: return 280
        }
    }

    fileprivate enum YScaleMode: Hashable { case linear, log }

    private var hasNonPositiveValues: Bool { chartPoints.contains { $0.value <= 0 } }

    private var effectiveScale: YScaleMode {
        (yScaleMode == .log && hasNonPositiveValues) ? .linear : yScaleMode
    }

    private func yAxisDomain(for mode: YScaleMode) -> ClosedRange<Double> {
        let values = chartPoints.map { $0.value }
        guard let minVal = values.min(), let maxVal = values.max() else { return 0...1 }
        switch mode {
        case .linear:
            if minVal == maxVal {
                let pad = max(abs(minVal) * 0.1, 0.1)
                return (minVal - pad)...(maxVal + pad)
            }
            let span = maxVal - minVal
            let pad = span * 0.05
            return (minVal - pad)...(maxVal + pad)
        case .log:
            let positives = values.filter { $0 > 0 }
            guard let minPos = positives.min(), let maxPos = positives.max() else { return 0.1...1 }
            let lower = minPos * 0.9
            let upper = maxPos * 1.1
            return max(lower, .leastNonzeroMagnitude)...max(upper, .leastNonzeroMagnitude)
        }
    }
}

// MARK: - Chart scale helper

extension View {
    @ViewBuilder
    fileprivate func applyYAxisScale(mode: MetricsChartSectionView.YScaleMode, domain: ClosedRange<Double>) -> some View
    {
        switch mode {
        case .linear: self.chartYScale(domain: domain, type: .linear)
        case .log: self.chartYScale(domain: domain, type: .log)
        }
    }
}

extension MetricsChartSectionView {
    fileprivate struct ChartPoint: Identifiable {
        let id: Int
        let step: Int
        let value: Double
        let name: String
    }

    fileprivate var chartPoints: [ChartPoint] {
        // Flatten pre-sorted series into a single array for LinePlot
        series.keys.sorted().flatMap { name in
            (series[name] ?? []).compactMap { m in
                guard let step = m.step else { return nil }
                return ChartPoint(id: m.id, step: step, value: m.value, name: name)
            }
        }
    }
}

private struct MetadataSectionView: View {
    let createdAt: Date
    let updatedAt: Date
    let experimentId: String
    let workspaceId: String?
    let tags: [String]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Metadata")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.custom.ctpText)

            HStack {
                Text("Created:")
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpSubtext0)
                Text(createdAt.conciseString)
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpText)
            }
            HStack {
                Text("Updated:")
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpSubtext0)
                Text(updatedAt.conciseString)
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpText)
            }
            HStack {
                Text("Experiment ID:")
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpSubtext0)
                Text(experimentId)
                    .font(.caption)
                    .foregroundStyle(Color.custom.ctpText)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if let workspaceId {
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

            if !tags.isEmpty {
                HStack(alignment: .center, spacing: 8) {
                    Text("Tags:")
                        .font(.caption)
                        .foregroundStyle(Color.custom.ctpSubtext0)
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            ForEach(tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption2)
                                    .foregroundStyle(Color.custom.ctpText)
                                    .lineLimit(1)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(
                                        RoundedRectangle(cornerRadius: 6)
                                            .fill(Color.custom.ctpSurface0.opacity(0.28))
                                    )
                            }
                        }
                        .padding(.vertical, 2)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }
}

// MARK: - Reusable UI

private struct SectionHeader: View {
    let title: String

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Color.custom.ctpText)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(Text(title))
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
