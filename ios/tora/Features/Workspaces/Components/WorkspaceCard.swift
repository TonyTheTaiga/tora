//
//  WorkspaceCard.swift
//  tora
//
//  Created by taiga on 7/21/25.
//

import SwiftUI

struct ExperimentRow: View {
    var experiment: Experiment

    var body: some View {
        Text(experiment.name)
            .font(.caption)
    }
}

struct WorkspaceCard: View {
    var workspace: Workspace
    var experiments: [Experiment]
    let onExperimentSelected: ((String) -> Void)
    @State private var isExpanded: Bool = false

    var body: some View {
        VStack(alignment: .leading) {
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Text(workspace.name)
                        .font(.headline)
                    Text("(\(experiments.count))")
                        .font(.headline)
                    Spacer()
                    Image(
                        systemName: isExpanded ? "chevron.up" : "chevron.down"
                    )
                }
            }
            .foregroundColor(Color.custom.ctpText)

            if isExpanded {
                Text("\(workspace.role)")
                    .font(.footnote)
                    .foregroundColor(getRoleColor(workspace.role))

                if !experiments.isEmpty {
                    Text("Experiments")
                        .font(.subheadline)

                    ForEach(experiments) { experiment in
                        Button(action: { onExperimentSelected(experiment.id) }) {
                            ExperimentRow(experiment: experiment)
                                .padding(.vertical, 2)
                        }
                        .foregroundColor(Color.custom.ctpBlue)
                    }
                } else {
                    Text("No experiments in this workspace.")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
            }
        }
    }

    private func getRoleColor(_ role: String) -> Color {
        switch role {
        case "OWNER":
            return .custom.ctpYellow
        case "ADMIN":
            return .custom.ctpRed
        case "VIEWER":
            return .custom.ctpGreen
        default:
            return .gray
        }
    }
}

#Preview(traits: .sizeThatFitsLayout) {
    WorkspaceCard(
        workspace:
            Workspace(
                id: "preview-id",
                name: "Some Workspace",
                description: "This is a sample workspace for previews.",
                createdAt: Date(),
                role: "OWNER"
            ),
        experiments: [
            Experiment(
                id: "exp-1",
                name: "MNIST Classifier",
                description: "Training a simple CNN to classify MNIST digits.",
                hyperparams: [
                    HyperParam(key: "learning_rate", value: .double(0.001)),
                    HyperParam(key: "epochs", value: .int(10)),
                    HyperParam(key: "batch_size", value: .int(64)),
                    HyperParam(key: "optimizer", value: .string("Adam")),
                ],
                tags: ["image-classification", "cnn", "pytorch"],
                createdAt: Date(),
                updatedAt: Date(),
                availableMetrics: ["accuracy", "loss", "validation_loss"],
                workspaceId: "preview-id",
                url: "https://example.com/exp-1"
            ),
            Experiment(
                id: "exp-2",
                name: "Sentiment Analysis",
                description:
                    "BERT-based model for sentiment analysis on movie reviews.",
                hyperparams: [
                    HyperParam(
                        key: "model_name",
                        value: .string("bert-base-uncased")
                    ),
                    HyperParam(key: "learning_rate", value: .double(0.00005)),
                    HyperParam(key: "max_length", value: .int(256)),
                ],
                tags: ["nlp", "sentiment", "bert", "transformers"],
                createdAt: Date(),
                updatedAt: Date(),
                availableMetrics: ["f1_score", "precision", "recall"],
                workspaceId: "preview-id",
                url: "https://example.com/exp-2"
            ),
        ],
        onExperimentSelected: { print("experimentId: \($0)") }
    )
}
