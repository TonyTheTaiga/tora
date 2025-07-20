import SwiftUI

extension Font {
    static func inter(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight)
    }

    static func interVariable(_ size: CGFloat) -> Font {
        .system(size: size)
    }

    static func interVariable(_ size: CGFloat, italic: Bool) -> Font {
        .system(size: size).italic(italic)
    }

    static func dynamicInter(
        _ size: CGFloat, weight: Font.Weight = .regular, relativeTo textStyle: Font.TextStyle = .body
    ) -> Font {
        .system(size: size, weight: weight, design: .default)
    }

    static func dynamicInterVariable(_ size: CGFloat, relativeTo textStyle: Font.TextStyle = .body)
        -> Font
    {
        .system(size: size, design: .default)
    }
}
