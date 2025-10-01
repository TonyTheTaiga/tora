import SwiftUI

struct ExperimentInfoPanel: View {
    let experiment: Experiment
    @Binding var isPresented: Bool

    private var tagColumns: [GridItem] {
        [GridItem(.adaptive(minimum: 96), spacing: 8, alignment: .leading)]
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    infoRow(title: "Created", value: experiment.createdAt.conciseString)
                    infoRow(title: "Updated", value: experiment.updatedAt.conciseString)
                    infoRow(title: "Experiment ID", value: experiment.id, truncationMode: .middle)

                    if let workspaceId = experiment.workspaceId {
                        infoRow(title: "Workspace ID", value: workspaceId, truncationMode: .middle)
                    }

                    if !experiment.tags.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Tags")
                                .font(.caption)
                                .foregroundStyle(Color.custom.ctpSubtext0)
                            LazyVGrid(columns: tagColumns, alignment: .leading, spacing: 8) {
                                ForEach(experiment.tags, id: \.self) { tag in
                                    Text(tag)
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(Color.custom.ctpText)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 6)
                                        .background(
                                            Capsule()
                                                .fill(Color.custom.ctpSurface0.opacity(0.2))
                                        )
                                }
                            }
                        }
                    }
                }
                .padding(.vertical, 16)
            }
            .padding(.horizontal)
            .background(Color.custom.ctpSheetBackground.ignoresSafeArea())
            .navigationTitle("Information")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { isPresented = false }
                }
            }
        }
    }

    private func infoRow(title: String, value: String, truncationMode: Text.TruncationMode = .tail) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.caption)
                .foregroundStyle(Color.custom.ctpSubtext0)
            Text(value)
                .font(.body.weight(.medium))
                .foregroundStyle(Color.custom.ctpText)
                .lineLimit(2)
                .truncationMode(truncationMode)
        }
    }
}

#Preview("Info Panel") {
    @Previewable @State var isPresented = true
    return ExperimentInfoPanel(
        experiment: Experiment(
            id: "exp_123",
            name: "Sample Experiment",
            description: "",
            hyperparams: [],
            tags: ["demo", "ios", "preview"],
            createdAt: .now,
            updatedAt: .now,
            workspaceId: "workspace_123",
            url: ""
        ),
        isPresented: $isPresented
    )
}
