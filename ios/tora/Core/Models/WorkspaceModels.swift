import Foundation

struct Workspace: Decodable, Identifiable, Equatable {
    var id: String
    var name: String
    var description: String?
    var createdAt: Date
    var role: String

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case description
        case createdAt
        case role
    }
}
