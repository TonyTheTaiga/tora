import SwiftUI

// MARK: - Legacy Shape (for backward compatibility if needed)
struct ToraIcon: Shape {
    func path(in rect: CGRect) -> Path {
        // Simplified placeholder - use asset-based ToraLogo view instead
        var path = Path()
        path.addRect(rect)
        return path
    }
}
