import Foundation

extension Date {
    var conciseString: String { formatted(date: .abbreviated, time: .shortened) }
}
