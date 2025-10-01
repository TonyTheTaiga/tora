import SwiftUI

struct ExperimentDetailView: View {
    @Binding var experiment: Experiment
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @EnvironmentObject private var experimentService: ExperimentService
    @StateObject private var viewModel = ExperimentDetailViewModel()
    @State private var isInfoPresented = false

    init(experiment: Binding<Experiment>) {
        self._experiment = experiment
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if let description = experiment.description, !description.isEmpty {
                    Text(description)
                        .font(.body)
                        .foregroundStyle(Color.custom.ctpSubtext0)
                }

                if !viewModel.metricsByName.isEmpty {
                    ExperimentMetricsSection(series: viewModel.metricsByName)
                }

                if !viewModel.results.isEmpty {
                    ExperimentResultsSection(results: viewModel.results)
                }

                if !experiment.hyperparams.isEmpty {
                    ExperimentHyperparamsSection(hyperparams: experiment.hyperparams)
                }
            }
            .padding()
        }
        .task { await viewModel.load(experimentID: experiment.id, service: experimentService) }
        .refreshable {
            await refreshExperiment()
            await viewModel.load(experimentID: experiment.id, service: experimentService)
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                HStack(spacing: 8) {
                    Text(experiment.name)
                        .font(.headline)
                        .multilineTextAlignment(.center)
                        .lineLimit(horizontalSizeClass == .regular ? 2 : 1)
                        .truncationMode(horizontalSizeClass == .regular ? .tail : .middle)
                        .minimumScaleFactor(0.9)

                    Button {
                        isInfoPresented = true
                    } label: {
                        Image(systemName: "info.circle")
                            .imageScale(.medium)
                    }
                    .accessibilityLabel("Show experiment information")
                }
            }
        }
        .sheet(isPresented: $isInfoPresented) {
            ExperimentInfoPanel(
                experiment: experiment,
                isPresented: $isInfoPresented
            )
            .presentationDetents([.medium, .large])
            .presentationDragIndicator(.visible)
        }
    }

    private func refreshExperiment() async {
        do {
            let updated = try await experimentService.get(experimentId: experiment.id)
            await MainActor.run { self.experiment = updated }
        } catch {
            // Intentionally ignore on pull-to-refresh errors so the sheet stays responsive.
        }
    }
}

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

    let authService = AuthService()
    let experimentService = ExperimentService(authService: authService)

    return ExperimentDetailView(experiment: $sample)
        .environmentObject(authService)
        .environmentObject(experimentService)
}
