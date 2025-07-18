import SwiftUI

extension Font {
    static func inter(_ size: CGFloat, weight: Font.Weight = .regular) -> Font {
        switch weight {
        case .bold:
            return .custom("Inter-Bold", size: size)
        case .medium:
            return .custom("Inter-Medium", size: size)
        case .regular:
            return .custom("Inter-Regular", size: size)
        default:
            return .custom("InterVariable", size: size)
        }
    }

    static func interVariable(_ size: CGFloat) -> Font {
        return .custom("InterVariable", size: size)
    }

    static func interVariable(_ size: CGFloat, italic: Bool) -> Font {
        if italic {
            return .custom("InterVariable-Italic", size: size)
        } else {
            return .custom("InterVariable", size: size)
        }
    }

    static func dynamicInter(
        _ size: CGFloat, weight: Font.Weight = .regular, relativeTo textStyle: Font.TextStyle = .body
    ) -> Font {
        switch weight {
        case .bold:
            return .custom("Inter-Bold", size: size, relativeTo: textStyle)
        case .medium:
            return .custom("Inter-Medium", size: size, relativeTo: textStyle)
        case .regular:
            return .custom("Inter-Regular", size: size, relativeTo: textStyle)
        default:
            return .custom("InterVariable", size: size, relativeTo: textStyle)
        }
    }

    static func dynamicInterVariable(_ size: CGFloat, relativeTo textStyle: Font.TextStyle = .body)
        -> Font
    {
        return .custom("InterVariable", size: size, relativeTo: textStyle)
    }
}
