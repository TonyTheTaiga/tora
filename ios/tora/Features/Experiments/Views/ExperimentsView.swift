import SwiftUI

struct ExperimentSelectorView: View {
    let experiments: [Experiment]
    let selectedExperiment: Experiment?
    let onExperimentSelected: (Experiment) -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationView {
            List(experiments) { experiment in
                Button(
                    action: {
                        onExperimentSelected(experiment)
                        dismiss()
                    },
                    label: {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(experiment.name)
                                .font(.headline)

                            if let description = experiment.description, !description.isEmpty {
                                Text(description)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .lineLimit(2)
                            }
                        }
                    }
                )
                .listRowBackground(
                    selectedExperiment?.id == experiment.id ? Color.blue.opacity(0.1) : Color.clear
                )
            }
            .navigationTitle("Select Experiment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark")
                    }
                }
            }
        }
    }
}

struct ExperimentsView: View {
    let initialExperimentId: String?
    @EnvironmentObject private var experimentService: ExperimentService
    @State private var allExperiments: [Experiment] = []
    @State private var selectedExperiment: Experiment?
    @State private var isLoadingExperiments = true
    @State private var isLoadingExperiment = false
    @State private var errorMessage: String?
    @State private var showingExperimentSelector = false

    init(experimentId: String? = nil) {
        self.initialExperimentId = experimentId
    }

    var body: some View {
        Group {
            if isLoadingExperiments {
                ProgressView("Loading experiments...")
            } else if let errorMessage = errorMessage {
                VStack {
                    Text("Error Loading Experiments")
                        .font(.headline)
                    Text(errorMessage)
                        .font(.caption)
                    Button("Try Again") {
                        fetchAllExperiments()
                    }
                }
            } else if allExperiments.isEmpty {
                Text("No Experiments")
            } else if let selectedExperiment = selectedExperiment {
                ExperimentContentView(experiment: selectedExperiment, isLoading: isLoadingExperiment)
            } else {
                VStack {
                    Text("Choose an Experiment")
                        .font(.headline)
                    Button("Choose Experiment") {
                        showingExperimentSelector = true
                    }
                }
            }
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                if !allExperiments.isEmpty {
                    Button(
                        action: {
                            showingExperimentSelector = true
                        },
                        label: {
                            HStack(spacing: 4) {
                                Text(selectedExperiment?.name.truncated(to: 20) ?? "Choose")
                                    .font(.headline)
                                Image(systemName: "chevron.down")
                                    .font(.caption)
                            }
                        })
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                Button(action: {
                    if let selectedExperiment = selectedExperiment {
                        fetchExperimentDetails(selectedExperiment.id)
                    } else {
                        fetchAllExperiments()
                    }
                }) {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(isLoadingExperiments || isLoadingExperiment)
            }
        }
        .onAppear {
            fetchAllExperiments()
        }
        .sheet(isPresented: $showingExperimentSelector) {
            ExperimentSelectorView(
                experiments: allExperiments,
                selectedExperiment: selectedExperiment,
                onExperimentSelected: { experiment in
                    selectExperiment(experiment)
                }
            )
        }
    }

    private func fetchAllExperiments() {
        isLoadingExperiments = true
        errorMessage = nil
        Task {
            do {
                let experiments = try await experimentService.listAll()
                await MainActor.run {
                    self.allExperiments = experiments
                    self.isLoadingExperiments = false

                    if let initialId = initialExperimentId,
                        let experiment = experiments.first(where: { $0.id == initialId })
                    {
                        selectExperiment(experiment)
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = error.localizedDescription
                    self.isLoadingExperiments = false
                }
            }
        }
    }

    private func selectExperiment(_ experiment: Experiment) {
        selectedExperiment = experiment
        fetchExperimentDetails(experiment.id)
    }

    private func fetchExperimentDetails(_ experimentId: String) {
        isLoadingExperiment = true
        Task {
            do {
                let experiment = try await experimentService.get(experimentId: experimentId)
                await MainActor.run {
                    self.selectedExperiment = experiment
                    self.isLoadingExperiment = false
                }
            } catch {
                await MainActor.run {
                    self.isLoadingExperiment = false
                }
            }
        }
    }
}

struct ExperimentContentView: View {
    let experiment: Experiment
    let isLoading: Bool

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if isLoading {
                    ProgressView("Loading experiment details...")
                } else {
                    if let description = experiment.description, !description.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Description")
                                .font(.headline)

                            Text(description)
                                .font(.body)
                        }
                    }

                    VStack(alignment: .leading, spacing: 12) {
                        Text("Metadata")
                            .font(.headline)

                        VStack(spacing: 8) {
                            HStack {
                                Text("Created")
                                    .font(.subheadline)
                                Spacer()
                                Text(formatDate(experiment.createdAt))
                                    .font(.subheadline)
                            }

                            HStack {
                                Text("Updated")
                                    .font(.subheadline)
                                Spacer()
                                Text(formatDate(experiment.updatedAt))
                                    .font(.subheadline)
                            }
                        }
                    }
                }
            }
            .padding()
        }
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

extension String {
    func truncated(to length: Int) -> String {
        if self.count <= length {
            return self
        } else {
            return String(self.prefix(length)) + "..."
        }
    }
}

#Preview {
    ExperimentsView()
        .environmentObject(ExperimentService(authService: AuthService.shared))
}
