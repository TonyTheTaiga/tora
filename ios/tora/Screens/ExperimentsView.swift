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

    // MARK: - Body

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(experiment.name)
                    .font(.title2)
                    .bold()

                if let description = experiment.description, !description.isEmpty {
                    Text(description)
                        .font(.body)
                }
            }
            .padding()
        }
        .refreshable { await onRefresh() }
    }
}

// MARK: - Preview

#Preview {
    @State var sample = Experiment(
        id: "exp_123",
        name: "Sample Experiment",
        description: "A simple example experiment description.",
        hyperparams: [],
        tags: [],
        createdAt: Date(),
        updatedAt: Date(),
        availableMetrics: [],
        workspaceId: nil,
        url: "https://example.com/exp_123"
    )

    return ExperimentsView(experiment: $sample)
}
