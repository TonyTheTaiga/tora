import SwiftUI

// MARK: - Layout System
// Responsive layout helpers inspired by web's mobile-first approach

// MARK: - Device Detection
public struct DeviceInfo {
    public static var isPhone: Bool {
        UIDevice.current.userInterfaceIdiom == .phone
    }

    public static var isPad: Bool {
        UIDevice.current.userInterfaceIdiom == .pad
    }

    public static var screenWidth: CGFloat {
        UIScreen.main.bounds.width
    }

    public static var screenHeight: CGFloat {
        UIScreen.main.bounds.height
    }

    public static var isCompact: Bool {
        screenWidth < 400
    }

    public static var isRegular: Bool {
        screenWidth >= 400
    }
}

// MARK: - Responsive Spacing
public struct ResponsiveSpacing {
    public static func horizontal() -> CGFloat {
        DeviceInfo.isPhone ? DesignSystem.Spacing.md : DesignSystem.Spacing.lg
    }

    public static func vertical() -> CGFloat {
        DeviceInfo.isPhone ? DesignSystem.Spacing.md : DesignSystem.Spacing.xl
    }

    public static func cardPadding() -> CGFloat {
        DeviceInfo.isCompact ? DesignSystem.Spacing.sm : DesignSystem.Spacing.md
    }

    public static func listItemPadding() -> CGFloat {
        DeviceInfo.isCompact ? DesignSystem.Spacing.sm : DesignSystem.Spacing.md
    }

    public static func sectionSpacing() -> CGFloat {
        DeviceInfo.isPhone ? DesignSystem.Spacing.lg : DesignSystem.Spacing.xl
    }
}

// MARK: - Responsive Typography
extension DesignSystem.Typography {
    public static func responsiveTitle1() -> Font {
        DeviceInfo.isPad
            ? Font.dynamicInter(32, weight: .bold, relativeTo: .largeTitle)
            : Font.dynamicInter(28, weight: .bold, relativeTo: .largeTitle)
    }

    public static func responsiveTitle2() -> Font {
        DeviceInfo.isPad
            ? Font.dynamicInter(26, weight: .bold, relativeTo: .title)
            : Font.dynamicInter(22, weight: .bold, relativeTo: .title)
    }

    public static func responsiveBody() -> Font {
        DeviceInfo.isCompact
            ? Font.dynamicInter(16, weight: .regular, relativeTo: .body)
            : Font.dynamicInter(17, weight: .regular, relativeTo: .body)
    }

    public static func responsiveCaption() -> Font {
        DeviceInfo.isCompact
            ? Font.dynamicInter(11, weight: .regular, relativeTo: .caption)
            : Font.dynamicInter(12, weight: .regular, relativeTo: .caption)
    }
}

// MARK: - Container Views
public struct ToraContainer<Content: View>: View {
    let content: Content
    let maxWidth: CGFloat?
    let horizontalPadding: CGFloat

    public init(
        maxWidth: CGFloat? = nil,
        horizontalPadding: CGFloat? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.content = content()
        self.maxWidth = maxWidth
        self.horizontalPadding = horizontalPadding ?? ResponsiveSpacing.horizontal()
    }

    public var body: some View {
        content
            .frame(maxWidth: maxWidth)
            .padding(.horizontal, horizontalPadding)
    }
}

// MARK: - Screen Container
public struct ToraScreen<Content: View>: View {
    let content: Content
    let backgroundColor: Color
    let showsScrollIndicators: Bool

    public init(
        backgroundColor: Color = DesignSystem.Colors.background,
        showsScrollIndicators: Bool = false,
        @ViewBuilder content: () -> Content
    ) {
        self.content = content()
        self.backgroundColor = backgroundColor
        self.showsScrollIndicators = showsScrollIndicators
    }

    public var body: some View {
        ScrollView(showsIndicators: showsScrollIndicators) {
            LazyVStack(spacing: 0) {
                content
            }
        }
        .background(backgroundColor)
    }
}

// MARK: - List Container
public struct ToraList<Data: RandomAccessCollection, Content: View>: View where Data.Element: Identifiable {
    let data: Data
    let content: (Data.Element, Int) -> Content
    let spacing: CGFloat
    let alternatingBackground: Bool

    public init(
        _ data: Data,
        spacing: CGFloat = 1,
        alternatingBackground: Bool = false,
        @ViewBuilder content: @escaping (Data.Element, Int) -> Content
    ) {
        self.data = data
        self.content = content
        self.spacing = spacing
        self.alternatingBackground = alternatingBackground
    }

    public var body: some View {
        LazyVStack(spacing: spacing) {
            ForEach(Array(data.enumerated()), id: \.element.id) { index, item in
                content(item, index)
            }
        }
    }
}

// MARK: - Grid Container
public struct ToraGrid<Data: RandomAccessCollection, Content: View>: View where Data.Element: Identifiable {
    let data: Data
    let columns: [GridItem]
    let content: (Data.Element) -> Content
    let spacing: CGFloat

    public init(
        _ data: Data,
        columns: Int = 2,
        spacing: CGFloat = DesignSystem.Spacing.md,
        @ViewBuilder content: @escaping (Data.Element) -> Content
    ) {
        self.data = data
        self.columns = Array(repeating: GridItem(.flexible(), spacing: spacing), count: columns)
        self.content = content
        self.spacing = spacing
    }

    public var body: some View {
        LazyVGrid(columns: columns, spacing: spacing) {
            ForEach(data) { item in
                content(item)
            }
        }
    }
}

// MARK: - Adaptive Stack
public struct AdaptiveStack<Content: View>: View {
    let horizontalAlignment: HorizontalAlignment
    let verticalAlignment: VerticalAlignment
    let spacing: CGFloat?
    let content: Content

    @Environment(\.horizontalSizeClass) var horizontalSizeClass

    public init(
        horizontalAlignment: HorizontalAlignment = .center,
        verticalAlignment: VerticalAlignment = .center,
        spacing: CGFloat? = nil,
        @ViewBuilder content: () -> Content
    ) {
        self.horizontalAlignment = horizontalAlignment
        self.verticalAlignment = verticalAlignment
        self.spacing = spacing
        self.content = content()
    }

    public var body: some View {
        Group {
            if horizontalSizeClass == .compact {
                VStack(alignment: horizontalAlignment, spacing: spacing) {
                    content
                }
            } else {
                HStack(alignment: verticalAlignment, spacing: spacing) {
                    content
                }
            }
        }
    }
}

// MARK: - Safe Area Container
public struct SafeAreaContainer<Content: View>: View {
    let content: Content
    let edges: Edge.Set

    public init(edges: Edge.Set = .all, @ViewBuilder content: () -> Content) {
        self.edges = edges
        self.content = content()
    }

    public var body: some View {
        content
            .padding(.top, edges.contains(.top) ? 0 : 0)
            .padding(.bottom, edges.contains(.bottom) ? 0 : 0)
            .padding(.leading, edges.contains(.leading) ? 0 : 0)
            .padding(.trailing, edges.contains(.trailing) ? 0 : 0)
            .ignoresSafeArea(.container, edges: edges)
    }
}

// MARK: - View Extensions for Layout
extension View {
    // Responsive padding
    func responsivePadding() -> some View {
        self.padding(.horizontal, ResponsiveSpacing.horizontal())
            .padding(.vertical, ResponsiveSpacing.vertical())
    }

    // Terminal-style section divider
    func terminalDivider() -> some View {
        self.overlay(
            Rectangle()
                .frame(height: 1)
                .foregroundColor(DesignSystem.Colors.divider),
            alignment: .bottom
        )
    }

    // Conditional modifier
    func `if`<Content: View>(_ condition: Bool, transform: (Self) -> Content) -> some View {
        Group {
            if condition {
                transform(self)
            } else {
                self
            }
        }
    }

    // Responsive frame
    func responsiveFrame(
        minWidth: CGFloat? = nil,
        idealWidth: CGFloat? = nil,
        maxWidth: CGFloat? = nil,
        minHeight: CGFloat? = nil,
        idealHeight: CGFloat? = nil,
        maxHeight: CGFloat? = nil,
        alignment: Alignment = .center
    ) -> some View {
        let adjustedMaxWidth = maxWidth.map { width in
            DeviceInfo.isCompact ? min(width, DeviceInfo.screenWidth * 0.9) : width
        }

        return self.frame(
            minWidth: minWidth,
            idealWidth: idealWidth,
            maxWidth: adjustedMaxWidth,
            minHeight: minHeight,
            idealHeight: idealHeight,
            maxHeight: maxHeight,
            alignment: alignment
        )
    }

    // Terminal-style border accent
    func terminalAccent(color: Color = DesignSystem.Colors.accent, width: CGFloat = 2) -> some View {
        self.overlay(
            Rectangle()
                .frame(width: width)
                .foregroundColor(color),
            alignment: .leading
        )
    }

    // Animated appearance
    func animatedAppearance(delay: Double = 0) -> some View {
        AnimatedAppearanceWrapper(delay: delay) {
            self
        }
    }
}

// MARK: - Animated Appearance Wrapper
private struct AnimatedAppearanceWrapper<Content: View>: View {
    let delay: Double
    let content: Content
    @State private var isVisible = false

    init(delay: Double, @ViewBuilder content: () -> Content) {
        self.delay = delay
        self.content = content()
    }

    var body: some View {
        content
            .opacity(isVisible ? 1 : 0)
            .offset(y: isVisible ? 0 : 20)
            .onAppear {
                withAnimation(DesignSystem.Animation.smooth.delay(delay)) {
                    isVisible = true
                }
            }
    }
}

// MARK: - Keyboard Responsive Modifier
struct KeyboardResponsive: ViewModifier {
    @State private var keyboardHeight: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .padding(.bottom, keyboardHeight)
            .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)) {
                notification in
                if let keyboardFrame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue {
                    withAnimation(DesignSystem.Animation.smooth) {
                        keyboardHeight = keyboardFrame.cgRectValue.height
                    }
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIResponder.keyboardWillHideNotification)) { _ in
                withAnimation(DesignSystem.Animation.smooth) {
                    keyboardHeight = 0
                }
            }
    }
}

extension View {
    func keyboardResponsive() -> some View {
        modifier(KeyboardResponsive())
    }
}
