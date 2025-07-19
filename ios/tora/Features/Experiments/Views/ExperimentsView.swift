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
                                .foregroundColor(.primary)

                            if let description = experiment.description, !description.isEmpty {
                                Text(description)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .lineLimit(2)
                            }

                            if let workspaceId = experiment.workspaceId {
                                Text("Workspace: \(workspaceId)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                )
                .buttonStyle(PlainButtonStyle())
                .listRowBackground(
                    selectedExperiment?.id == experiment.id ? Color.blue.opacity(0.1) : Color.clear
                )
            }
            .navigationTitle("Select Experiment")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    ToraToolbarButton(systemImage: "xmark") {
                        dismiss()
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
                VStack(spacing: 16) {
                    ProgressView()
                        .scaleEffect(1.2)
                    Text("Loading experiments...")
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let errorMessage = errorMessage {
                ContentUnavailableView {
                    Label("Error Loading Experiments", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(errorMessage)
                } actions: {
                    ToraButton("Try Again", style: .primary) {
                        fetchAllExperiments()
                    }
                }
            } else if allExperiments.isEmpty {
                ContentUnavailableView {
                    Label("No Experiments", systemImage: "flask")
                } description: {
                    Text("No experiments are available across your workspaces.")
                }
            } else if let selectedExperiment = selectedExperiment {
                ExperimentContentView(experiment: selectedExperiment, isLoading: isLoadingExperiment)
            } else {
                ContentUnavailableView {
                    Label("Choose an Experiment", systemImage: "flask")
                } description: {
                    Text("Select an experiment from the header to view its details.")
                } actions: {
                    ToraButton("Choose Experiment", style: .primary) {
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
                            .foregroundColor(.primary)
                        })
                }
            }

            ToolbarItem(placement: .navigationBarTrailing) {
                ToraToolbarButton(systemImage: "arrow.clockwise") {
                    if let selectedExperiment = selectedExperiment {
                        fetchExperimentDetails(selectedExperiment.id)
                    } else {
                        fetchAllExperiments()
                    }
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
                    // Keep the basic experiment info even if detailed fetch fails
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
                    VStack(spacing: 16) {
                        ProgressView()
                            .scaleEffect(1.2)
                        Text("Loading experiment details...")
                            .foregroundColor(.secondary)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding(.top, 100)
                } else {
                    // Description Section
                    if let description = experiment.description, !description.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Description")
                                .font(.headline)
                                .foregroundColor(.primary)

                            Text(description)
                                .font(.body)
                                .foregroundColor(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    // Metadata Section
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Metadata")
                            .font(.headline)
                            .foregroundColor(.primary)

                        VStack(spacing: 8) {
                            HStack {
                                Text("Created")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                    .foregroundColor(.primary)
                                Spacer()
                                Text(formatDate(experiment.createdAt))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }

                            HStack {
                                Text("Updated")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                    .foregroundColor(.primary)
                                Spacer()
                                Text(formatDate(experiment.updatedAt))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }

                            if let workspaceId = experiment.workspaceId {
                                HStack {
                                    Text("Workspace")
                                        .font(.subheadline)
                                        .fontWeight(.medium)
                                        .foregroundColor(.primary)
                                    Spacer()
                                    Text(workspaceId)
                                        .font(.subheadline)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    .padding()
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))

                    // Tags Section
                    if !experiment.tags.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Tags")
                                .font(.headline)
                                .foregroundColor(.primary)

                            LazyVGrid(
                                columns: [
                                    GridItem(.adaptive(minimum: 80), spacing: 8)
                                ], spacing: 8
                            ) {
                                ForEach(experiment.tags, id: \.self) { tag in
                                    Text(tag)
                                        .font(.subheadline)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 6)
                                        .background(.blue.opacity(0.1))
                                        .foregroundColor(.blue)
                                        .clipShape(Capsule())
                                }
                            }
                        }
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    // Hyperparameters Section
                    if !experiment.hyperparams.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Hyperparameters")
                                .font(.headline)
                                .foregroundColor(.primary)

                            VStack(spacing: 8) {
                                ForEach(experiment.hyperparams, id: \.key) { hyperparam in
                                    HStack {
                                        Text(hyperparam.key)
                                            .font(.subheadline)
                                            .fontWeight(.medium)
                                            .foregroundColor(.primary)
                                        Spacer()
                                        Text(hyperparam.value.displayValue)
                                            .font(.subheadline)
                                            .foregroundColor(.secondary)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(.secondary.opacity(0.1))
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
                    }

                    // Metrics Section
                    if !experiment.availableMetrics.isEmpty {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Available Metrics")
                                .font(.headline)
                                .foregroundColor(.primary)

                            LazyVGrid(
                                columns: [
                                    GridItem(.adaptive(minimum: 100), spacing: 8)
                                ], spacing: 8
                            ) {
                                ForEach(experiment.availableMetrics, id: \.self) { metric in
                                    Text(metric)
                                        .font(.subheadline)
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 6)
                                        .background(.green.opacity(0.1))
                                        .foregroundColor(.green)
                                        .clipShape(Capsule())
                                }
                            }
                        }
                        .padding()
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
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
        .preferredColorScheme(.light)
}
