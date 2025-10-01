import Charts
import SwiftUI

struct ExperimentMetricsSection: View {
    let series: [String: [Metric]]
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @State private var yScaleMode: YScaleMode = .linear

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                ExperimentSectionHeader(title: "Metrics")
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

    enum YScaleMode: Hashable { case linear, log }

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

    private var chartPoints: [ChartPoint] {
        series.keys.sorted().flatMap { name in
            (series[name] ?? []).compactMap { metric in
                guard let step = metric.step else { return nil }
                return ChartPoint(id: metric.id, step: step, value: metric.value, name: name)
            }
        }
    }

    private struct ChartPoint: Identifiable {
        let id: Int
        let step: Int
        let value: Double
        let name: String
    }
}

struct ExperimentResultsSection: View {
    let results: [ExperimentDetailViewModel.ResultItem]

    private var grid: [GridItem] { [GridItem(.adaptive(minimum: 140), spacing: 12, alignment: .leading)] }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ExperimentSectionHeader(title: "Results")
            LazyVGrid(columns: grid, alignment: .leading, spacing: 12) {
                ForEach(results) { item in
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

struct ExperimentHyperparamsSection: View {
    let hyperparams: [HyperParam]

    private var grid: [GridItem] { [GridItem(.adaptive(minimum: 140), spacing: 12, alignment: .leading)] }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ExperimentSectionHeader(title: "Hyperparameters")

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

struct ExperimentSectionHeader: View {
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

extension View {
    @ViewBuilder
    fileprivate func applyYAxisScale(mode: ExperimentMetricsSection.YScaleMode, domain: ClosedRange<Double>)
        -> some View
    {
        switch mode {
        case .linear:
            chartYScale(domain: domain, type: .linear)
        case .log:
            chartYScale(domain: domain, type: .log)
        }
    }
}
