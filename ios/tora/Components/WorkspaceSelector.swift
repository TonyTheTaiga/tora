import SwiftUI

// MARK: - Workspace Picker Modal (Centered)

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
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("Select Workspace")
                    .font(.headline)
                Spacer()
                Button(action: onClose) {
                    Image(systemName: "xmark")
                        .foregroundStyle(.secondary)
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 16)
            .padding(.top, 14)
            .padding(.bottom, 8)

            // Search field
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
            .padding(.horizontal, 16)
            .padding(.bottom, 10)

            Divider()

            // List
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(ordered) { ws in
                        Button(action: {
                            onWorkspaceSelected(ws)
                            onClose()
                        }) {
                            HStack {
                                Text(ws.name)
                                    .font(.body)
                                    .lineLimit(1)
                                    .truncationMode(.tail)
                                    .foregroundStyle(Color.custom.ctpText)
                                Spacer()
                                if ws.id == selectedWorkspace?.id {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(Color.custom.ctpBlue)
                                }
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .contentShape(Rectangle())
                        }
                        Divider()
                    }
                }
            }
            .frame(maxHeight: 360)
        }
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(Color.custom.ctpMantle)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(Color.custom.ctpOverlay2.opacity(0.7), lineWidth: 1)
        )
        .frame(maxWidth: 520)
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
