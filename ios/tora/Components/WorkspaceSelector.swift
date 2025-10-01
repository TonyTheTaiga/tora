import SwiftUI

// MARK: - Workspace Picker Sheet

struct WorkspacePickerModal: View {
    let workspaces: [Workspace]
    let selectedWorkspace: Workspace?
    let onWorkspaceSelected: (Workspace) -> Void
    let onClose: () -> Void
    @State private var query: String = ""

    private var filtered: [Workspace] {
        if query.isEmpty { return workspaces }
        return workspaces.filter { $0.name.localizedCaseInsensitiveContains(query) }
    }

    private var ordered: [Workspace] {
        guard let selected = selectedWorkspace else { return filtered }
        var result: [Workspace] = []
        result.append(selected)
        result += filtered.filter { $0.id != selected.id }
        return result
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(Color.custom.ctpSubtext0)
                    TextField("Search workspaces", text: $query)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled(true)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.custom.ctpSurface0)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.custom.ctpOverlay1, lineWidth: 1)
                )

                if ordered.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "magnifyingglass")
                            .imageScale(.large)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                        Text("No workspaces match your search")
                            .font(.subheadline)
                            .foregroundStyle(Color.custom.ctpSubtext0)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List {
                        ForEach(ordered) { ws in
                            Button {
                                onWorkspaceSelected(ws)
                                onClose()
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(ws.name)
                                            .font(.body)
                                            .foregroundStyle(Color.custom.ctpText)
                                            .lineLimit(1)

                                        if let description = ws.description, !description.isEmpty {
                                            Text(description)
                                                .font(.caption)
                                                .foregroundStyle(Color.custom.ctpSubtext0)
                                                .lineLimit(2)
                                        }
                                    }
                                    Spacer()
                                    if ws.id == selectedWorkspace?.id {
                                        Image(systemName: "checkmark")
                                            .foregroundStyle(Color.custom.ctpBlue)
                                    }
                                }
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                }
            }
            .padding(.horizontal)
            .padding(.top, 20)
            .padding(.bottom, 12)
            .background(Color.custom.ctpSheetBackground.ignoresSafeArea())
            .navigationTitle("Select Workspace")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { onClose() }
                }
            }
        }
    }
}

// MARK: - Workspace Selector Pill

struct WorkspaceSelector: View {
    let selectedWorkspace: Workspace?
    let onOpenPicker: () -> Void

    var body: some View {
        Button(action: onOpenPicker) {
            let title = selectedWorkspace?.name ?? "Select a Workspace"
            Text(title)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.tail)
                .foregroundStyle(selectedWorkspace == nil ? Color.custom.ctpSubtext0 : Color.custom.ctpText)
                .padding(.horizontal, 14)
                .padding(.vertical, 8)
                .background(
                    Capsule().fill(Color.custom.ctpMantle)
                )
                .overlay(
                    Capsule()
                        .stroke(Color.custom.ctpOverlay2, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
        .frame(maxWidth: 260, alignment: .leading)
    }
}

// MARK: - Preview

#Preview(traits: .sizeThatFitsLayout) {
    let ws = [
        Workspace(id: "w1", name: "Core ML", description: "Core team experiments", createdAt: .now, role: "OWNER"),
        Workspace(
            id: "w2", name: "NLP Super Long Workspace Name That Should Truncate Nicely", description: "Language models",
            createdAt: .now, role: "VIEWER"),
    ]
    return VStack(spacing: 20) {
        WorkspaceSelector(
            selectedWorkspace: ws.first,
            onOpenPicker: {}
        )
        WorkspacePickerModal(
            workspaces: ws,
            selectedWorkspace: ws.first,
            onWorkspaceSelected: { _ in },
            onClose: {}
        )
        .frame(height: 420)
    }
}
