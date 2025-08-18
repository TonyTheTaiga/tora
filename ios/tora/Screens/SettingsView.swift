import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var authService: AuthService
    @AppStorage("appTheme") private var selectedThemeRaw: String = AppTheme.system.rawValue

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 12) {
                    Text("Settings")
                        .font(.title2)
                        .fontWeight(.semibold)
                    Spacer()
                    ToraButton(
                        "logout",
                        size: .small,
                        backgroundColor: Color.custom.ctpSurface0.opacity(0.20),
                        borderColor: Color.custom.ctpSurface0.opacity(0.30),
                        textColor: Color.custom.ctpRed,
                        systemImage: "rectangle.portrait.and.arrow.right"
                    ) {
                        Task { logout() }
                    }
                    .accessibilityLabel("Sign out")
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)

                HStack(spacing: 12) {
                    Text("Theme")
                        .font(.body)
                    Spacer()
                    Picker("", selection: $selectedThemeRaw) {
                        ForEach(AppTheme.allCases) { theme in
                            Text(theme.displayName).tag(theme.rawValue)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .fixedSize()
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 12)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private func logout() {
        try? authService.logout()
    }
}

// Preview scaffolding
#Preview {
    SettingsView()
}
